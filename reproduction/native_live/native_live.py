"""Protocol-bound native DBpedia live experiment; archived sources are read-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RELEASE = WORKSPACE
sys.path[:0] = [str(RELEASE/'reproduction/completion'),
    str(RELEASE/'reproduction/study'), str(RELEASE/'reproduction/runtime/external_study'),
    str(RELEASE/'reproduction/runtime/greedy_count_profile'), str(RELEASE/'code'),
    str(WORKSPACE/'reproduction/sharp_value')]
import common
import native_dbpedia
import run_study as study

PAIRS = (('agent', 'person'), ('person', 'athlete'))
PREDICATES = ('agent', 'person', 'athlete')
METHODS = ('batched_right', 'adaptive_continuous', 'separate', 'joint')
SEEDS = (101, 202)
YAML = common.SOURCE/'configs/predicates/dbpedia.yaml'
DECODE = dict(n_predict=4, temperature=0, seed=17,
              grammar='root ::= "true" | "false"', cache_prompt=False, stream=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def questions():
    result = {}
    for block in YAML.read_text(encoding='utf-8').split('\n- id: ')[1:]:
        name = block.splitlines()[0].strip()
        if name in PREDICATES:
            match = re.search(r"^  langex: '(.*)'$", block, re.M)
            assert match, name
            result[name] = match.group(1).replace('{text}', 'The quoted article')
    assert set(result) == set(PREDICATES)
    return result


QUESTIONS = questions()


def prompt(text, predicate):
    return ('<|im_start|>system\nClassify the quoted article. Answer exactly true or false.<|im_end|>\n'
        '<|im_start|>user\nStatement: '+QUESTIONS[predicate]+'.\nArticle: '+
        json.dumps(text, ensure_ascii=False)+'\nIs the statement true?\nAnswer:<|im_end|>\n'
        '<|im_start|>assistant\n')


def load_cases(rows):
    assert tuple(common.DBPEDIA_PAIRS[:2]) == PAIRS
    cases = [c for c in native_dbpedia.cases_for() if
        tuple(p.rsplit('/', 1)[-1] for p in (c.predicate_left, c.predicate_right)) in PAIRS]
    assert len(cases) == 4 and all(c.n == 10000 for c in cases)
    if rows == 5000:
        return cases
    if not 1 <= rows <= 5000:
        raise ValueError('Rows must be in 1..5000')
    selected = sorted(sorted(range(5000), key=lambda j:
        hashlib.sha256(('certis-native-live-v1:'+cases[0].row_ids[j]).encode()).hexdigest())[:rows])
    out = []
    for case in cases:
        old = selected+[j+5000 for j in selected]
        mapping = {j:i for i,j in enumerate(old)}
        left = tuple(tuple(mapping[j] for j in ids if j in mapping) for ids in case.key_left)
        right = tuple(tuple(mapping[j] for j in ids if j in mapping) for ids in case.key_right)
        # Empty declared keys remain in the target and are handled by the original runtime.
        out.append(common.Case(dataset=case.dataset, predicate_left=case.predicate_left,
            predicate_right=case.predicate_right, variant=case.variant,
            ids=tuple(case.ids[j] for j in old), side=case.side[old], key_names=case.key_names,
            key_left=left, key_right=right, scores=case.scores[old],
            row_ids=tuple(case.row_ids[j] for j in old), texts=tuple(case.texts[j] for j in old),
            metadata=dict(case.metadata, target_version='sha256:'+jsha((case.target_version, selected)),
                live_rows=rows, selection='fixed SHA256 certis-native-live-v1 subset'),
            _tape_key=case._tape_key))
    return out


class Client:
    def __init__(self, port):
        self.base = f'http://127.0.0.1:{port}'

    def api(self, path, data=None, timeout=300):
        request = urllib.request.Request(self.base+path,
            data=None if data is None else json.dumps(data).encode('utf-8'),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def infer(self, prompts):
        started = time.perf_counter()
        result = self.api('/completion', dict(prompt=prompts, **DECODE))
        elapsed = time.perf_counter()-started
        if isinstance(result, dict):
            result = [result]
        assert len(result) == len(prompts)
        rows = []
        for item in result:
            assert not item.get('truncated'), 'Prompt exceeds context; no truncation allowed'
            assert item.get('content', '').strip() in ('true', 'false'), item
            assert item.get('timings', {}).get('cache_n', 0) == 0, 'Unexpected prompt cache'
            rows.append(dict(label=int(item['content'].strip() == 'true'),
                tokens_in=item['tokens_evaluated'], tokens_out=item['tokens_predicted'],
                timings=item.get('timings', {}), raw=item))
        return dict(wall_seconds=elapsed, responses=rows)


def server_command(args):
    return [str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1',
        '--port', str(args.port), '-ngl', '99', '--ctx-size', str(args.context), '--parallel', '1',
        '--no-cont-batching', '--batch-size', '2048', '--ubatch-size', '512',
        '--flash-attn', 'on', '--cache-ram', '0', '--no-cache-prompt', '--no-webui', '--threads', '4']


@contextmanager
def owned_server(args, out):
    command = server_command(args)
    started = time.perf_counter()
    log = (out/'server.log').open('ab')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    study.save(out/'owned_process.json', dict(pid=process.pid, executable=args.server.name, command=command))
    client = Client(args.port)
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Owned server exited; inspect server.log')
            try:
                if client.api('/health', timeout=1).get('status') == 'ok':
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Server startup timeout')
        study.save(out/'server_props.json', client.api('/props'))
        study.save(out/'startup.json', dict(seconds=time.perf_counter()-started))
        yield client
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()


def prepare(args, cases, out):
    source_files = [Path(__file__), HERE/'PROTOCOL.md', YAML,
        RELEASE/'reproduction/completion/native_dbpedia.py', *study.source_paths(),
        RELEASE/'code/piecewise_numpy_profile.py', RELEASE/'code/piecewise_count_profile.py']
    cost_sql = HERE/'cost_sql.py'
    if cost_sql.exists():
        source_files.append(cost_sql)
    if args.stage == 'run' and not cost_sql.exists():
        raise FileNotFoundError('The returned-first SQL harness is required before running')
    wrapper = WORKSPACE/'reproduction/sharp_value/sharp_wrapper.py'
    if wrapper.exists():
        source_files.append(wrapper)
    if args.stage == 'run' and not wrapper.exists():
        raise FileNotFoundError('The fixed separate/joint wrapper must exist before running')
    half = cases[0].n//2
    prompts = [prompt(cases[0].texts[j], p) for p in PREDICATES for j in range(half)]
    info = dict(protocol_sha256=sha(HERE/'PROTOCOL.md'),
        sources={str(p.relative_to(WORKSPACE)):sha(p) for p in dict.fromkeys(source_files)},
        data_sha256=sha(common.SOURCE/'data/datasets/dbpedia.parquet'),
        rows_per_side=half, roster=list(cases[0].row_ids[:half]),
        pairs=PAIRS, predicates=QUESTIONS, cases=[c.case_id for c in cases],
        target_versions={c.case_id:c.target_version for c in cases},
        canonical_prompt_sha256=jsha(prompts), per_prompt_sha256=[jsha(p) for p in prompts],
        methods=METHODS, seeds=SEEDS, audit_budget=args.audit_budget,
        execution_environment=dict(python=sys.version, platform=platform.platform(),
            numpy=np.__version__, pandas=pd.__version__),
        recall='9/10', delta='1/20', server_command=server_command(args), decoding=DECODE,
        server_sha256=sha(args.server), runtime_dll_sha256={p.name:sha(p) for p in args.server.parent.glob('*.dll')},
        model_sha256={p.name:sha(p) for p in args.model.parent.glob(args.model.name.split('-00001')[0]+'*.gguf')},
        prompt_truncation='forbidden', slots=1, continuous_batching=False, logical_batch=128,
        original_70b_labels_used_for_policy=False, selection='all rows' if half == 5000 else 'fixed hash subset')
    binding = jsha(info)
    manifest = dict(binding=binding, **info)
    path = out/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == json.loads(json.dumps(manifest))
    else:
        study.save(path, manifest)
        for file in dict.fromkeys(source_files):
            dest = out/'source_versions'/(sha(file)+file.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(file.read_bytes())
    return binding, prompts


def token_check(client, prompts, context, out):
    path = out/'prompt_tokens.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['canonical_prompt_sha256'] == jsha(prompts)
    else:
        started = time.perf_counter()
        counts = []
        for i, p in enumerate(prompts):
            result = client.api('/tokenize', dict(content=p, add_special=True, parse_special=True))
            counts.append(len(result['tokens']))
            if (i+1) % 3000 == 0:
                print(json.dumps(dict(stage='tokenize', completed=i+1, total=len(prompts))), flush=True)
        result = dict(canonical_prompt_sha256=jsha(prompts), counts=counts,
            max_input_tokens=max(counts), median=float(np.median(counts)),
            seconds=time.perf_counter()-started, context=context, output_allowance=4)
        study.save(path, result)
    assert result['max_input_tokens']+4 <= context, 'Increase context before any oracle labels; never truncate'
    return result


def preflight(client, cases, out, binding, token_counts):
    warmup = client.infer([prompt(cases[0].texts[j], PREDICATES[j%3]) for j in range(8)])
    study.save(out/'warmup.json', warmup)
    half = cases[0].n//2
    chosen = sorted(range(half), key=lambda j:
        hashlib.sha256(('certis-native-preflight-v1:'+cases[0].row_ids[j]).encode()).hexdigest())[:128]
    sample = [prompt(cases[0].texts[j], PREDICATES[i%3]) for i,j in enumerate(chosen)]
    baseline = client.infer(sample)
    expected = [r['label'] for r in baseline['responses']]
    checks = []
    for cap in (1, 8, 128):
        order = list(reversed(range(len(sample))))
        observed = {}; batches = []
        for start in range(0, len(sample), cap):
            ids = order[start:start+cap]
            measured = client.infer([sample[j] for j in ids])
            observed.update((j,row['label']) for j,row in zip(ids, measured['responses']))
            batches.append(dict(ids=ids, **measured))
        checks.append(dict(cap=cap, batches=batches,
            mismatches=[j for j in range(len(sample)) if observed[j] != expected[j]]))
        study.save(out/'repeatability.json', dict(binding=binding, selected_row_indices=chosen,
            baseline=baseline, checks=checks))
        assert not checks[-1]['mismatches'], 'Oracle repeatability failed; preserve failure'
    per_prompt = baseline['wall_seconds']/len(sample)
    rows = half
    # Bound counts without examining labels. Every method queries each identity at most once.
    estimate = dict(binding=binding, status='pass', fixed_sample_size=len(sample),
        measured_seconds_per_prompt=per_prompt, sample_input_tokens_mean=float(np.mean([
            r['tokens_in'] for r in baseline['responses']])),
        all_input_tokens_mean=float(np.mean(token_counts['counts'])),
        reference_requests=3*rows, executions=32, max_paid_query_requests=32*2*rows,
        reference_seconds_estimate=3*rows*per_prompt,
        query_seconds_upper_estimate=32*2*rows*per_prompt,
        note='Throughput extrapolation only; query count bound does not use labels. Planning and SQL add cost.')
    study.save(out/'preflight_summary.json', estimate)
    print(json.dumps(dict(stage='preflight_complete', **estimate)), flush=True)
    return estimate


def dense_reference(client, cases, out, binding):
    reference_started = time.perf_counter()
    case = cases[0]; half = case.n//2
    references = {}; frames = []
    for predicate in PREDICATES:
        rows = []
        for start in range(0, half, 128):
            path = out/'reference_batches'/f'{predicate}_{start:05d}.json'
            ids = list(range(start, min(start+128, half)))
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                assert record['binding'] == binding
            else:
                record = dict(binding=binding, ids=ids, **client.infer([
                    prompt(case.texts[j], predicate) for j in ids]))
                study.save(path, record)
            assert record['ids'] == ids
            rows.extend(record['responses'])
            if start%1024 == 0:
                print(json.dumps(dict(stage='reference', predicate=predicate, completed=len(rows))), flush=True)
        references[predicate] = rows
        frames.extend(dict(row_id=case.row_ids[j], predicate_id='dbpedia/'+predicate,
            label=row['label']) for j,row in enumerate(rows))
    parquet = out/'fixed_oracle.parquet'
    pd.DataFrame(frames).to_parquet(parquet, index=False)
    for c in cases:
        pl,pr = [p.rsplit('/', 1)[-1] for p in (c.predicate_left,c.predicate_right)]
        response = references[pl]+references[pr]
        arrays = [np.array([r[k] for r in response], dtype=np.int64)
            for k in ('label','tokens_in','tokens_out')]
        zeros = np.zeros(c.n, dtype=np.int64)
        for array in arrays+[zeros]:
            array.setflags(write=False)
        key = 'native-live:'+binding+':'+pl+':'+pr
        common._TAPES[key] = common._Tape(*arrays, zeros, zeros)
        c._tape_key = key
        c.metadata = dict(c.metadata, oracle_parquet=str(parquet.resolve()),
            oracle_contract_version='qwen2.5-7b-q4km:'+binding,
            oracle_source='fresh complete local reference; original 70B labels not used')
    inference_seconds = sum(json.loads(p.read_text())['wall_seconds']
        for p in (out/'reference_batches').glob('*.json'))
    study.save(out/'reference_binding.json', dict(parquet=parquet.name, sha256=sha(parquet),
        requests=len(frames), wall_seconds=inference_seconds,
        inference_seconds=inference_seconds, total_seconds=time.perf_counter()-reference_started,
        note='Dense reference is evaluation-only; total includes persistence and tape/parquet setup.'))


def execute(client, case, method, seed, budget, out):
    import sharp_wrapper
    from cost_sql import returned_first_evaluate
    instances = []
    original = study.BatchOracle
    original_evaluate = common.evaluate

    class LiveOracle(common.Oracle):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.batches = []; self.monitor_seconds = 0.
            instances.append(self)

        def acquire(self, ids, phase):
            assert phase in common.PHASES
            ids = [j for j in common._ids(ids, self.case.n) if j not in self.facts]
            new = {}
            for start in range(0, len(ids), 128):
                selected = ids[start:start+128]
                ps = [self.case.predicate_left if self.case.side[j] == 0 else self.case.predicate_right
                    for j in selected]
                prompts = [prompt(self.case.texts[j], p.rsplit('/',1)[-1]) for j,p in zip(selected,ps)]
                measured = client.infer(prompts)
                batch = dict(phase=phase, ids=selected, size=len(selected),
                    prompt_sha256=[jsha(p) for p in prompts], **measured)
                self.batches.append(batch)
                for j,row in zip(selected, measured['responses']):
                    y = row['label']
                    monitor = time.perf_counter()
                    expected = int(self._validation_labels()[j])
                    if y != expected:
                        study.save(out/'oracle_disagreement.json', dict(case_id=case.case_id,
                            method=method, seed=seed, identity=j, live=y, reference=expected, batch=batch))
                        raise AssertionError('Live label differs from fixed reference; retained failure')
                    self.monitor_seconds += time.perf_counter()-monitor
                    self.facts[j] = new[j] = y
                    self.ledger.append(dict(id=j, identity=self.case.ids[j], phase=phase, label=y,
                        tokens_in=row['tokens_in'], tokens_out=row['tokens_out']))
            return new

    case.close_sql()
    study.BatchOracle = LiveOracle
    common.evaluate = returned_first_evaluate
    started = time.perf_counter()
    try:
        if method == 'batched_right':
            result = study.batched_exact(case, method)
        elif method == 'adaptive_continuous':
            result = study.marginal(case, seed, budget, method)
        else:
            result = sharp_wrapper.execute(case, seed, budget, method=method,
                oracle_class=LiveOracle, instrument=False)
    finally:
        study.BatchOracle = original
        common.evaluate = original_evaluate
    outer = time.perf_counter()-started
    assert len(instances) == 1, len(instances)
    oracle = instances[0]
    assert not oracle.helper_ledger, 'Unmeasured helper cost is forbidden in live experiment'
    assert [j for b in oracle.batches for j in b['ids']] == [r['id'] for r in oracle.ledger]
    result['batches'] = oracle.batches
    output = result['output']
    assert output['sql_evaluation_order'] == 'returned-before-reference'
    assert output['cold_returned_first']
    inference = sum(b['wall_seconds'] for b in oracle.batches)
    sql = sum(output[k] for k in ('sql_preprocessing_seconds','fact_ingest_seconds',
        'returned_materialization_seconds'))
    raw_complete = result['timings']['algorithm_seconds']+sql
    complete = raw_complete-oracle.monitor_seconds
    coverage = all(r['lower'] <= sum(int(oracle._validation_labels()[j]) for j in r['ids']) <= r['upper']
        for r in result.get('regions',[]))
    assert not coverage or 10*output['lost_count'] <= output['full_count']
    assert output['bag_subset']
    row = dict(case_id=case.case_id, key=case.variant, method=method, seed=seed,
        audit_budget=0 if method=='batched_right' else budget, requests=len(oracle.ledger),
        batches=len(oracle.batches), inference_seconds=inference, complete_seconds=complete,
        complete_with_inline_monitor_seconds=raw_complete, inline_oracle_validation_seconds=oracle.monitor_seconds,
        local_seconds=complete-inference, returned_sql_seconds=sql, outer_seconds=outer,
        excluded_verification_seconds=outer-complete, coverage=coverage,
        conflict=result.get('conflict_freeze',False), loss=output['loss'],
        full_count=output['full_count'], returned_count=output['returned_count'],
        tokens_in=sum(r['tokens_in'] for r in oracle.ledger), tokens_out=sum(r['tokens_out'] for r in oracle.ledger),
        inference_by_phase={p:sum(b['wall_seconds'] for b in oracle.batches if b['phase']==p)
            for p in common.PHASES}, all_live_labels_equal_fixed_reference=True,
        live_outputs_used_for_policy=True, helper_requests=0)
    return dict(status='pass', row=row, result=result)


def run(args):
    os.chdir(RELEASE)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); cases = load_cases(args.rows)
    study.save(out/'corpus_loading.json', dict(seconds=time.perf_counter()-started))
    if args.stage == 'describe':
        desc = dict(rows_per_side=args.rows, pairs=PAIRS, predicates=QUESTIONS,
            queries=[dict(case_id=c.case_id, keys=len(c.key_names), nonempty_keys=sum(bool(a) and bool(b)
                for a,b in zip(c.key_left,c.key_right)), max_key_size=max(map(len,c.key_left))) for c in cases],
            dense_reference_requests=3*args.rows, executions=32,
            maximum_paid_query_requests=64*args.rows)
        study.save(out/'description.json', desc); print(json.dumps(desc), flush=True)
        return
    binding, prompts = prepare(args, cases, out)
    with owned_server(args, out) as client:
        tokens = token_check(client, prompts, args.context, out)
        preflight(client, cases, out, binding, tokens)
        if args.stage == 'preflight':
            return
        dense_reference(client, cases, out, binding)
        rows = []
        for repetition, seed in enumerate(SEEDS):
            for ci,case in enumerate(cases):
                shift = (repetition+ci)%len(METHODS)
                for method in METHODS[shift:]+METHODS[:shift]:
                    path = out/'raw'/f'{case.case_id}__{method}__s{seed}.json.gz'
                    if path.exists():
                        record = study.read(path); assert record['binding'] == binding
                    else:
                        record = dict(binding=binding, repetition=repetition,
                            **execute(client,case,method,seed,args.audit_budget,out))
                        study.save(path,record,True)
                    rows.append(record['row'])
                    study.save(out/'progress.json',dict(completed=len(rows),expected=32,rows=rows))
                    print(json.dumps(dict(stage='query',completed=len(rows),**record['row'])),flush=True)
                case.close_sql()
        study.save(out/'summary.json',dict(status='pass',binding=binding,expected=32,verified=len(rows),
            query_count=4,repetitions=2,audit_seeds=SEEDS,rows=rows,
            live_outputs_used_for_policy=True,fixed_reference_disagreements=0))
    for case in cases:
        case.close_sql()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('describe','preflight','run'))
    p.add_argument('--server',type=Path,default=Path('PATH_TO_LLAMA_SERVER'))
    p.add_argument('--model',type=Path,default=Path('PATH_TO_GGUF'))
    p.add_argument('--port',type=int,default=18087)
    p.add_argument('--context',type=int,default=1024)
    p.add_argument('--rows',type=int,default=5000)
    p.add_argument('--audit-budget',type=int,default=500)
    p.add_argument('--output',type=Path,default=RELEASE/'outputs/native_live')
    run(p.parse_args())
