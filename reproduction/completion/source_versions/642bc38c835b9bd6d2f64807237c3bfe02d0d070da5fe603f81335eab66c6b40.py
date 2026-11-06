"""Execute natural queries using paid, live labels from a local fixed oracle."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'reproduction/study'),
               str(ROOT/'reproduction/runtime/external_study'),
               str(ROOT/'reproduction/runtime/greedy_count_profile'), str(ROOT/'code')]
import run_study as study
import common
import cached_greedy

PAIRS = (('sent_positive', 'sent_negative'), ('approval', 'disapproval'))
METHODS = ('batched_right', 'adaptive_continuous', 'joint_greedy_cached')
QUESTIONS = {'sent_positive': 'positive sentiment', 'sent_negative': 'negative sentiment',
             'approval': 'approval', 'disapproval': 'disapproval'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt(text, predicate):
    return ('<|im_start|>system\nClassify the quoted comment. Answer exactly true or false.<|im_end|>\n'
            '<|im_start|>user\nDoes the comment express '+QUESTIONS[predicate]+'?\nComment: '+
            json.dumps(text, ensure_ascii=False)+'\nAnswer:<|im_end|>\n<|im_start|>assistant\n')


class Client:
    def __init__(self, port):
        self.base = f'http://127.0.0.1:{port}'

    def api(self, path, data=None, timeout=300):
        request = urllib.request.Request(self.base+path,
            data=None if data is None else json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def infer(self, prompts):
        started = time.perf_counter()
        result = self.api('/completion', dict(prompt=prompts, n_predict=4, temperature=0,
            seed=17, grammar='root ::= "true" | "false"', cache_prompt=False, stream=False))
        elapsed = time.perf_counter()-started
        if isinstance(result, dict):
            result = [result]
        assert len(result) == len(prompts)
        rows = []
        for item in result:
            assert not item.get('truncated'), 'Prompt exceeds configured context'
            assert item.get('content', '').strip() in ('true', 'false'), item.get('content')
            assert item.get('timings', {}).get('cache_n', 0) == 0, 'Unexpected prompt cache reuse'
            rows.append(dict(label=int(item['content'].strip() == 'true'),
                tokens_in=item['tokens_evaluated'], tokens_out=item['tokens_predicted'],
                timings=item.get('timings', {})))
        return dict(wall_seconds=elapsed, responses=rows)


def dense_reference(client, cases, out, binding):
    case = cases[0]
    half = case.n//2
    frames = []
    references = {}
    for predicate in [p for pair in PAIRS for p in pair]:
        rows = []
        for start in range(0, half, 128):
            path = out/'reference_batches'/f'{predicate}_{start:05d}.json'
            ids = list(range(start, min(start+128, half)))
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                assert record['binding'] == binding
            else:
                record = dict(binding=binding, ids=ids,
                    **client.infer([prompt(case.texts[j], predicate) for j in ids]))
                study.save(path, record)
            assert record['ids'] == ids
            rows.extend(record['responses'])
            if start % 1024 == 0:
                print(json.dumps(dict(stage='reference', predicate=predicate, completed=len(rows))), flush=True)
        references[predicate] = rows
        frames.extend(dict(row_id=case.row_ids[j], predicate_id='goemotions/'+predicate,
                           label=row['label']) for j, row in enumerate(rows))
    parquet = out/'fixed_oracle.parquet'
    pd.DataFrame(frames).to_parquet(parquet, index=False)
    for case in cases:
        pl, pr = (p.rsplit('/', 1)[-1] for p in (case.predicate_left, case.predicate_right))
        rows = references[pl]+references[pr]
        arrays = [np.array([r[k] for r in rows], dtype=np.int64)
                  for k in ('label', 'tokens_in', 'tokens_out')]
        zeros = np.zeros(case.n, dtype=np.int64)
        for array in arrays+[zeros]:
            array.setflags(write=False)
        key = 'local-fixed:'+binding+':'+pl+':'+pr
        common._TAPES[key] = common._Tape(*arrays, zeros, zeros)
        case._tape_key = key
        case.metadata = dict(case.metadata, oracle_parquet=str(parquet.resolve()),
            oracle_contract_version='qwen2.5-7b-q4km:'+binding,
            oracle_source='fresh local complete reference; original 70B labels not used')
    return parquet


def execute(client, case, method, seed, out):
    instances = []
    original = study.BatchOracle

    class LiveOracle(common.Oracle):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batches = []
            instances.append(self)

        def acquire(self, ids, phase):
            ids = [j for j in common._ids(ids, self.case.n) if j not in self.facts]
            new = {}
            for start in range(0, len(ids), 128):
                selected = ids[start:start+128]
                predicates = [self.case.predicate_left if self.case.side[j] == 0
                              else self.case.predicate_right for j in selected]
                measured = client.infer([prompt(self.case.texts[j], p.rsplit('/', 1)[-1])
                                         for j, p in zip(selected, predicates)])
                batch = dict(phase=phase, ids=selected, size=len(selected), **measured)
                self.batches.append(batch)
                # Model outputs, never reference labels, drive the facts below.
                for j, row in zip(selected, measured['responses']):
                    y = row['label']
                    if y != self._validation_labels()[j]:
                        study.save(out/'oracle_disagreement.json', dict(case_id=case.case_id,
                            method=method, seed=seed, identity=j, live=y,
                            reference=int(self._validation_labels()[j]), batch=batch))
                        raise AssertionError('Live label disagrees with fixed reference; retained failure')
                    self.facts[j] = new[j] = y
                    self.ledger.append(dict(id=j, identity=self.case.ids[j], phase=phase,
                        label=y, tokens_in=row['tokens_in'], tokens_out=row['tokens_out']))
            return new

    study.BatchOracle = LiveOracle
    cached_greedy.BatchOracle = LiveOracle
    case.close_sql()
    started = time.perf_counter()
    try:
        if method == 'batched_right':
            result = study.batched_exact(case, method)
        elif method == 'adaptive_continuous':
            result = study.marginal(case, seed, 500, method)
        else:
            result = cached_greedy.execute(case, seed, 500)
    finally:
        study.BatchOracle = original
        cached_greedy.BatchOracle = original
    outer_seconds = time.perf_counter()-started
    assert len(instances) == 1
    oracle = instances[0]
    assert [j for batch in oracle.batches for j in batch['ids']] == [r['id'] for r in oracle.ledger]
    result['batches'] = oracle.batches
    inference = sum(b['wall_seconds'] for b in oracle.batches)
    output = result['output']
    sql_seconds = sum(output[k] for k in ('sql_preprocessing_seconds', 'fact_ingest_seconds',
                                         'returned_materialization_seconds'))
    complete = result['timings']['algorithm_seconds']+sql_seconds
    coverage = all(r['lower'] <= sum(int(oracle._validation_labels()[j]) for j in r['ids']) <= r['upper']
                   for r in result.get('regions', []))
    assert not coverage or 10*output['lost_count'] <= output['full_count']
    assert output['bag_subset']
    row = dict(case_id=case.case_id, method=method, seed=seed, audit_budget=0 if method=='batched_right' else 500,
        requests=len(oracle.ledger), batches=len(oracle.batches),
        inference_seconds=inference, complete_seconds=complete, local_seconds=complete-inference,
        returned_sql_seconds=sql_seconds, outer_seconds=outer_seconds,
        excluded_verification_seconds=outer_seconds-complete,
        coverage=coverage, conflict=result.get('conflict_freeze', False),
        loss=output['loss'], full_count=output['full_count'], returned_count=output['returned_count'],
        tokens_in=sum(r['tokens_in'] for r in oracle.ledger), tokens_out=sum(r['tokens_out'] for r in oracle.ledger),
        inference_by_phase={p:sum(b['wall_seconds'] for b in oracle.batches if b['phase']==p)
                            for p in ('audit', 'execution')},
        all_live_labels_equal_fixed_reference=True, live_outputs_used_for_policy=True)
    return dict(status='pass', row=row, result=result)


def fixed_subset(cases):
    selected=sorted(sorted(range(cases[0].n//2),key=lambda j:
        hashlib.sha256(('certis-live-v2:'+cases[0].row_ids[j]).encode()).hexdigest())[:2048])
    result=[]
    for case in cases:
        old=selected+[j+case.n//2 for j in selected]
        mapping={j:i for i,j in enumerate(old)}
        key_left=tuple(tuple(mapping[j] for j in ids if j in mapping) for ids in case.key_left)
        key_right=tuple(tuple(mapping[j] for j in ids if j in mapping) for ids in case.key_right)
        result.append(common.Case(dataset=case.dataset,predicate_left=case.predicate_left,
            predicate_right=case.predicate_right,variant=case.variant,
            ids=tuple(case.ids[j] for j in old),side=case.side[old],key_names=case.key_names,
            key_left=key_left,key_right=key_right,scores=case.scores[old],
            row_ids=tuple(case.row_ids[j] for j in old),texts=tuple(case.texts[j] for j in old),
            metadata=dict(case.metadata,target_version='sha256:'+hashlib.sha256(
                (case.target_version+json.dumps(selected)).encode()).hexdigest(),
                live_rows=2048,selection='SHA256 fixed subset; original natural keys retained')))
    return result


def repeatability(client,cases,out):
    sample=[prompt(cases[0].texts[j],list(QUESTIONS)[j%4]) for j in range(128)]
    baseline=client.infer(sample)
    expected=[r['label'] for r in baseline['responses']]
    checks=[]
    for cap in (1,8,128):
        order=list(reversed(range(128)))
        observed={}
        elapsed=0.
        for start in range(0,128,cap):
            ids=order[start:start+cap]
            r=client.infer([sample[j] for j in ids]);elapsed+=r['wall_seconds']
            observed.update((j,item['label']) for j,item in zip(ids,r['responses']))
        check=dict(cap=cap,wall_seconds=elapsed,
            mismatches=[j for j in range(128) if observed[j]!=expected[j]])
        checks.append(check)
        study.save(out/'repeatability.json',dict(checks=checks,baseline=baseline))
        assert not check['mismatches'],'Canonical-slot repeatability failed'
    return checks


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [case for case in study.cases_for('native')
             if tuple(p.rsplit('/', 1)[-1] for p in (case.predicate_left, case.predicate_right)) in PAIRS]
    assert len(cases) == 4
    cases=fixed_subset(cases)
    command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1',
        '--port', str(args.port), '-ngl', '99', '--ctx-size', '1024', '--parallel', '1','--no-cont-batching',
        '--batch-size', '2048', '--ubatch-size', '512', '--flash-attn', 'on', '--cache-ram', '0',
        '--no-cache-prompt', '--no-webui', '--threads', '4']
    binding_info = dict(driver_sha256=digest(__file__), protocol_sha256=digest(HERE/'PROTOCOL.md'),
        runtime_addendum_sha256=digest(HERE/'RUNTIME_ADDENDUM.md'),
        server_sha256=digest(args.server), model_shards={p.name:digest(p)
            for p in args.model.parent.glob(args.model.name.split('-00001')[0]+'*.gguf')},
        cases=[c.case_id for c in cases], seeds=[101, 202], methods=METHODS,
        target_versions={c.case_id:c.target_version for c in cases}, prompt_questions=QUESTIONS,
        original_70b_labels_used_for_policy=False, live_outputs_used_for_policy=True,
        live_comments=2048,server_slots=1,continuous_batching=False,logical_acquisition_batch=128)
    binding = hashlib.sha256(json.dumps(binding_info, sort_keys=True).encode()).hexdigest()
    manifest = dict(binding=binding, **binding_info)
    if (out/'manifest.json').exists():
        assert json.loads(json.dumps(manifest)) == json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    else:
        study.save(out/'manifest.json', manifest)
    log = (out/'server.log').open('ab')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    client = Client(args.port)
    study.save(out/'owned_process.json', dict(pid=process.pid, executable=args.server.name))
    rows = []
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Local server exited; inspect server.log')
            try:
                if client.api('/health', timeout=1).get('status') == 'ok':
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Local server startup timed out')
        study.save(out/'server_props.json', client.api('/props'))
        study.save(out/'warmup.json', client.infer([prompt(cases[0].texts[j], 'sent_positive') for j in range(128)]))
        repeatability(client,cases,out)
        reference = dense_reference(client, cases, out, binding)
        study.save(out/'reference_binding.json', dict(parquet=reference.name, sha256=digest(reference)))
        for repetition, seed in enumerate((101, 202)):
            for ci, case in enumerate(cases):
                shift = (repetition+ci) % len(METHODS)
                for method in METHODS[shift:]+METHODS[:shift]:
                    key = f'{case.case_id}__{method}__s{seed}'
                    path = out/'raw'/(key+'.json.gz')
                    if path.exists():
                        record = study.read(path)
                        assert record['binding'] == binding
                    else:
                        record = dict(binding=binding, repetition=repetition,
                            **execute(client, case, method, seed, out))
                        study.save(path, record, True)
                    rows.append(record['row'])
                    study.save(out/'progress.json', dict(completed=len(rows), expected=24, rows=rows))
                    print(json.dumps(dict(stage='query', completed=len(rows), **record['row'])), flush=True)
                case.close_sql()
        study.save(out/'summary.json', dict(status='pass', binding=binding, expected=24, verified=len(rows),
            query_count=4, repetitions=2, audit_seeds=[101,202], rows=rows,
            live_outputs_used_for_policy=True, fixed_reference_disagreements=0))
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()
        for case in cases:
            case.close_sql()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18086)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/closed_loop')
    run(parser.parse_args())
