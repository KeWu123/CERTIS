"""Actual offline GPU batch timing for fixed, fully charged native traces."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'reproduction/runtime/external_study'),
              str(ROOT/'reproduction/runtime/greedy_count_profile'),str(ROOT/'code')]
from run_study import cases_for, save, read, sha

SERVER=None
MODEL=None
CASE='goemotions_native__sent_positive__sent_negative__subreddit'
PORT=18080
BASE=f'http://127.0.0.1:{PORT}'
METHODS=('batched_right','adaptive_continuous','joint_greedy')


def api(path,data=None,timeout=180):
    payload=None if data is None else json.dumps(data).encode('utf-8')
    req=urllib.request.Request(BASE+path,data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)


def prompt(case,j):
    predicate=case.predicate_left if int(case.side[j])==0 else case.predicate_right
    question='Does the comment express '+('positive sentiment' if predicate.endswith('sent_positive') else 'negative sentiment')+'?'
    return ('<|im_start|>system\nClassify the quoted comment. Answer exactly true or false.<|im_end|>\n'
            '<|im_start|>user\n'+question+'\nComment: '+json.dumps(case.texts[j],ensure_ascii=False)+
            '\nAnswer:<|im_end|>\n<|im_start|>assistant\n')


def infer(prompts):
    started=time.perf_counter()
    response=api('/completion',dict(prompt=prompts,n_predict=4,temperature=0,seed=17,
        grammar='root ::= "true" | "false"',cache_prompt=False,stream=False))
    elapsed=time.perf_counter()-started
    if isinstance(response,dict): response=[response]
    assert len(response)==len(prompts),(len(response),len(prompts))
    assert all(not r.get('truncated') and r.get('content','').strip() in ('true','false') for r in response)
    assert all(r.get('timings',{}).get('cache_n',0)==0 for r in response),'Unexpected KV reuse'
    return dict(wall_seconds=elapsed,size=len(prompts),responses=[dict(
        label=r['content'].strip(),tokens_evaluated=r.get('tokens_evaluated'),tokens_predicted=r.get('tokens_predicted'),
        timings=r.get('timings'),truncated=r.get('truncated'),stop_type=r.get('stop_type')) for r in response])


def run(args):
    global SERVER, MODEL
    SERVER=args.server.resolve(); MODEL=args.model.resolve()
    out=HERE/args.output; out.mkdir(parents=True,exist_ok=True)
    cases=cases_for('native'); case=next(c for c in cases if c.case_id==CASE)
    records={}
    for method in METHODS:
        seed,budget=(0,0) if method.startswith('batched') else (101,500)
        key=f'{CASE}__{method}__s{seed}__b{budget}'
        p=HERE/'native_v1/raw'/(key+'.json.gz')
        records[method]=(p,read(p))
    command=[str(SERVER),'-m',str(MODEL),'--host','127.0.0.1','--port',str(PORT),'-ngl','99',
        '--ctx-size','65536','--parallel','128','--batch-size','2048','--ubatch-size','512',
        '--flash-attn','on','--cache-ram','0','--no-cache-prompt','--no-webui','--threads','4']
    manifest=dict(command=[SERVER.name]+command[1:2]+[MODEL.name]+command[3:],model_shards={p.name:sha(p) for p in MODEL.parent.glob('qwen2.5-7b*.gguf')},
        server_sha256=sha(SERVER),source_sha256=sha(__file__),protocol_sha256=sha(HERE/'LIVE_PROTOCOL.md'),
        traces={m:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for m,(p,r) in records.items()},
        correctness_oracle='Llama-3.3-70B archived tape; unchanged',timing_model='Qwen2.5-7B-Instruct Q4_K_M',
        gpu='RTX 4090 D',prompt_template=prompt(case,0).replace(json.dumps(case.texts[0],ensure_ascii=False),'<COMMENT_JSON>'),
        live_outputs_used_for_policy=False)
    save(out/'manifest.json',manifest)
    log=(out/'server.log').open('wb')
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    save(out/'owned_process.json',dict(pid=process.pid,executable=SERVER.name))
    try:
        for attempt in range(90):
            if process.poll() is not None: raise RuntimeError('Local server exited; see server.log')
            try:
                health=api('/health',timeout=1)
                if health.get('status')=='ok': break
            except Exception: pass
            time.sleep(1)
        else: raise RuntimeError('Local model did not become ready')
        save(out/'server_props.json',api('/props'))
        control_ids=records['batched_right'][1]['result']['batches'][0]['ids']
        prompts=[prompt(case,j) for j in control_ids]
        save(out/'warmup.json',infer(prompts))
        probes=[]
        for rep in range(3):
            caps=(1,8,32,128)
            caps=caps[rep:]+caps[:rep]
            for cap in caps:
                p=out/'probes'/f'r{rep}_b{cap}.json'
                started=time.perf_counter(); chunks=[]
                for pos in range(0,len(prompts),cap): chunks.append(infer(prompts[pos:pos+cap]))
                row=dict(repetition=rep,client_cap=cap,wall_seconds=time.perf_counter()-started,chunks=chunks)
                save(p,row); probes.append(row)
                print(json.dumps(dict(stage='probe',rep=rep,cap=cap,seconds=row['wall_seconds'])),flush=True)
        save(out/'probes.json',probes)
        batches={m:r['result']['batches'] for m,(p,r) in records.items()}
        measured=defaultdict(list)
        # Interleave one acquisition round per policy to spread thermal/order effects.
        for index in range(max(map(len,batches.values()))):
            for method in METHODS:
                if index>=len(batches[method]): continue
                batch=batches[method][index]
                result=infer([prompt(case,j) for j in batch['ids']])
                result.update(method=method,index=index,phase=batch['phase'],ids=batch['ids'])
                save(out/'batches'/f'{method}_{index:03d}.json',result)
                measured[method].append(result)
            if index%5==0:
                save(out/'progress.json',{m:dict(rounds=len(v),wall_seconds=sum(x['wall_seconds'] for x in v)) for m,v in measured.items()})
                print(json.dumps(dict(stage='trace',index=index,rounds={m:len(v) for m,v in measured.items()})),flush=True)
        totals={m:dict(rounds=len(v),requests=sum(x['size'] for x in v),
            measured_inference_seconds=sum(x['wall_seconds'] for x in v),
            prompt_tokens=sum(r['tokens_evaluated'] for x in v for r in x['responses']),
            generated_tokens=sum(r['tokens_predicted'] for x in v for r in x['responses'])) for m,v in measured.items()}
        save(out/'summary.json',dict(status='pass',methods=totals,full_trace_repetitions=1,
            probe_repetitions=3,cache_prompt=False,all_prompts_untruncated=True,
            interpretation='Measured fixed-trace latency on a separate local model; live outputs did not choose the traces.'))
        print(json.dumps(totals),flush=True)
    finally:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
        for c in cases: c.close_sql()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--server',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',default='../../outputs/local_timing')
    run(p.parse_args())
