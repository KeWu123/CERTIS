"""AUTHOR checks for the NEW strict post-execution verifier.

These tests are not a continuation of the frozen independent audit. They
create their own temporary Parquet tapes and no historical raw record is
changed or exhaustively verified. Run with python -X utf8 -B.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import copy
import hashlib
import json
import tempfile
import time
import numpy as np
import pandas as pd
from common import make_tiny_case, Oracle, evaluate
from direct_arms import make_regions
from strict_result_validation import load_pinned_reference, strict_check_record, StrictValidationError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(folder, nside=3, *, empty=False, name='strict_fixture'):
    n = 0 if empty else 2*nside
    labels = [int(j % 3 != 1) for j in range(n)]
    tokens = {key: np.arange(n, dtype=np.int64)*scale+offset for key, scale, offset in
              [('oracle_input_tokens', 3, 5), ('oracle_output_tokens', 2, 7),
               ('helper_input_tokens', 5, 11), ('helper_output_tokens', 7, 13)]}
    # Empty joins keep the n=9996 SQL fixture small, while all labels/charges
    # are still independently joined and checked for every identity.
    left = ((),) if nside > 10 or empty else (tuple(range(nside)),)
    right = ((),) if nside > 10 or empty else (tuple(range(nside, n)),)
    case = make_tiny_case(side=[0]*(n//2)+[1]*(n//2), key_left=left, key_right=right,
                          truth=labels, tokens=tokens, name=name)
    case.row_ids = tuple(f'row{j}' for j in range(n//2))*2
    predicates = [case.predicate_left]*(n//2)+[case.predicate_right]*(n//2)
    oracle = pd.DataFrame({'row_id': pd.Series(case.row_ids, dtype='string'),
                           'predicate_id': pd.Series(predicates, dtype='string'),
                           'label': np.asarray(labels, dtype=np.int8),
                           'tokens_in': tokens['oracle_input_tokens'],
                           'tokens_out': tokens['oracle_output_tokens']})
    helper = oracle.copy(); helper['label'] = 1-helper['label']
    helper['score'] = case.scores
    helper['tokens_in'] = tokens['helper_input_tokens']; helper['tokens_out'] = tokens['helper_output_tokens']
    op = folder/(name+'__oracle-llama.parquet'); hp = folder/(name+'__helper.parquet')
    oracle.to_parquet(op, index=False); helper.to_parquet(hp, index=False)
    case.metadata.update(oracle_parquet=str(op), helper_parquet=str(hp),
                         source_hashes={op.name: sha(op), hp.name: sha(hp)})
    return case, oracle, helper


def make_record(case, *, seed=101, budget=1, helper=True, exact=False):
    oracle = Oracle(case)
    if helper:
        oracle.use_scores(range(case.n))
    if exact:
        regions = None
    else:
        _, regions, _ = make_regions(case, oracle, seed, budget)
    oracle.acquire(range(case.n), 'execution')
    method = 'exact_scan' if exact else 'count_allocation_feedback_audit_degree'
    result = dict(method=method, summary=oracle.summary(), facts=dict(oracle.facts),
                  oracle_ledger=list(oracle.ledger), helper_ledger=list(oracle.helper_ledger),
                  output=evaluate(case, oracle.facts), discarded=[], allowance='1/10', delta='1/20')
    if exact:
        result['no_audit'] = True
    else:
        result.update(seed=seed, audit_budget=budget, regions=regions)
    top_seed, top_budget = (0, 0) if exact else (seed, budget)
    top_method = 'scan' if exact else method
    key = f'{case.case_id}__{top_method}__s{top_seed}__b{top_budget}'
    record = dict(key=key, status='pass', case_id=case.case_id, dataset=case.dataset,
                  variant=case.variant, method=top_method, seed=top_seed, budget=top_budget,
                  source_id='test-source', supplement_source_id='test-supplement',
                  target_version=case.target_version, oracle_version=case.oracle_version,
                  metadata=copy.deepcopy(case.metadata), result=result)
    args = dict(record_path=HERE/(key+'.json.gz'), expected_key=key, expected_method=top_method,
                expected_seed=top_seed, expected_budget=top_budget, expected_source_id='test-source',
                expected_supplement_source_id='test-supplement')
    return record, args


def main():
    started = time.perf_counter(); positives=[]; rejected=[]
    with tempfile.TemporaryDirectory(prefix='strict_verifier_fixture_', dir=HERE) as directory:
        folder=Path(directory).resolve()
        # Cleanup is restricted to this newly created verified study child.
        assert HERE.resolve() in folder.parents
        case, oracle_frame, helper_frame = fixture(folder)
        reference=load_pinned_reference(case)
        assert len(set(case.row_ids)) == case.n//2 and len(reference.labels)==case.n
        record,args=make_record(case)
        check=strict_check_record(record,case,reference=reference,matched_result=record['result'],**args)
        positives.append(dict(name='small_audited_helper_and_oracle',identities=case.n,**check))

        def reject(name, mutate, code, *, matched=False, change_args=None):
            bad=copy.deepcopy(record);options=dict(args)
            mutate(bad)
            if change_args:change_args(options)
            try:
                strict_check_record(bad,case,reference=reference,
                                    matched_result=record['result'] if matched else None,**options)
            except StrictValidationError as exc:
                assert exc.code==code,(name,exc.code,code,str(exc))
                rejected.append(dict(name=name,code=exc.code))
            else:raise AssertionError('Mutation accepted: '+name)

        def mutate_selfconsistent_tokens(r):
            data=r['result'];summary=data['summary']
            for entry in data['oracle_ledger']:entry['tokens_in']+=12345
            summary['oracle_tokens_in']+=12345*len(data['oracle_ledger'])
            for phase in ('audit','cut','execution'):
                summary['phase_tokens'][phase]['tokens_in']+=12345*summary['phase_counts'][phase]
        reject('oracle_tokens_plus_all_selfconsistent_totals',mutate_selfconsistent_tokens,'ledger.oracle_tokens')
        def mutate_helper(r):
            r['result']['helper_ledger'][0]['tokens_out']+=77
            r['result']['summary']['helper_tokens_out']+=77
        reject('helper_tokens_plus_selfconsistent_total',mutate_helper,'ledger.helper_tokens')
        reject('phase_tokens_only',lambda r:r['result']['summary']['phase_tokens']['execution'].__setitem__('tokens_out',999),'summary.phase_tokens')
        reject('phase_counts_only',lambda r:r['result']['summary']['phase_counts'].__setitem__('cut',1),'summary.phase_counts')
        for field,value in [('key','wrong-key'),('case_id','wrong-case'),('method','wrong-method'),
                            ('seed',202),('budget',2),('source_id','wrong-source'),('supplement_source_id','wrong-supplement')]:
            reject('cell_'+field,lambda r,f=field,v=value:r.__setitem__(f,v),'cell.'+field)
        reject('filename',lambda r:None,'cell.filename',change_args=lambda a:a.__setitem__('record_path',HERE/'wrong.json.gz'))
        reject('result_method',lambda r:r['result'].__setitem__('method','wrong-method'),'cell.result_method')
        reject('result_seed',lambda r:r['result'].__setitem__('seed',202),'cell.result_seed')
        reject('result_budget',lambda r:r['result'].__setitem__('audit_budget',2),'cell.result_budget')
        reject('oracle_identity',lambda r:r['result']['oracle_ledger'][0].__setitem__('identity','wrong'),'ledger.oracle_identity')
        reject('oracle_label',lambda r:r['result']['oracle_ledger'][0].__setitem__('label',1-r['result']['oracle_ledger'][0]['label']),'ledger.oracle_label')
        reject('oracle_phase',lambda r:r['result']['oracle_ledger'][0].__setitem__('phase','free'),'ledger.oracle_phase')
        reject('oracle_duplicate',lambda r:r['result']['oracle_ledger'].append(copy.deepcopy(r['result']['oracle_ledger'][0])),'ledger.oracle_duplicate')
        reject('oracle_out_of_range',lambda r:r['result']['oracle_ledger'][0].__setitem__('id',case.n),'ledger.oracle_id')
        reject('helper_identity',lambda r:r['result']['helper_ledger'][0].__setitem__('identity','wrong'),'ledger.helper_identity')
        reject('helper_duplicate',lambda r:r['result']['helper_ledger'].append(copy.deepcopy(r['result']['helper_ledger'][0])),'ledger.helper_duplicate')
        reject('facts_changed',lambda r:r['result']['facts'].__setitem__(0,1-r['result']['facts'][0]),'facts.ledger')
        reject('wrong_request_total',lambda r:r['result']['summary'].__setitem__('oracle_requests',999),'summary.oracle_requests')
        reject('region_against_matched_original',lambda r:r['result']['regions'][0].__setitem__('upper',999),'evidence.regions_match',matched=True)
        reject('region_intrinsic_inversion',lambda r:r['result']['regions'][0].__setitem__('upper',999),'evidence.regions_values')
        reject('regions_removed',lambda r:r['result'].pop('regions'),'evidence.regions_missing')
        def swap_audit(r):
            rows=r['result']['oracle_ledger'];rows[0],rows[1]=rows[1],rows[0]
        reject('ordered_audit_swap',swap_audit,'evidence.audit_order')
        # A supplied matched original is also bound even if the current audit
        # is correct: changing the comparator must reject independently.
        wrong_match=copy.deepcopy(record['result']);wrong_match['oracle_ledger'][0]['identity']='wrong-match'
        try:strict_check_record(record,case,reference=reference,matched_result=wrong_match,**args)
        except StrictValidationError as exc:
            assert exc.code=='evidence.audit_match';rejected.append(dict(name='wrong_matched_ordered_audit',code=exc.code))
        else:raise AssertionError('Wrong comparator audit accepted')

        # Valid JSON string fact keys, explicit method wrapper aliases, no
        # supplement original records, and seed0/budget0 point conventions.
        json_record=json.loads(json.dumps(record));strict_check_record(json_record,case,reference=reference,**args)
        positives.append(dict(name='json_fact_keys',identities=case.n))
        alias=copy.deepcopy(record);alias['method']='wrapper';alias['key']='wrapper-cell'
        alias_args={**args,'expected_method':'wrapper','expected_key':'wrapper-cell','record_path':HERE/'wrapper-cell.json.gz','result_method_aliases':(record['result']['method'],)}
        strict_check_record(alias,case,reference=reference,**alias_args);positives.append(dict(name='explicit_wrapper_alias'))
        original=copy.deepcopy(record);original.pop('supplement_source_id')
        strict_check_record(original,case,reference=reference,**{**args,'expected_supplement_source_id':None})
        positives.append(dict(name='original_no_supplement'))
        point,p_args=make_record(case,seed=101,budget=0)
        point['seed']=0;point['key']=point['key'].replace('__s101__b0','__s0__b0')
        p_args.update(expected_key=point['key'],record_path=HERE/(point['key']+'.json.gz'),expected_seed=0)
        strict_check_record(point,case,reference=reference,seed0_control_seed=101,**p_args)
        positives.append(dict(name='explicit_seed0_point_inner_seed101'))
        try:strict_check_record(point,case,reference=reference,**p_args)
        except StrictValidationError as exc:
            assert exc.code=='cell.result_seed';rejected.append(dict(name='undeclared_seed0_exception',code=exc.code))
        else:raise AssertionError('Undeclared seed exception accepted')
        try:strict_check_record(record,case,reference=reference,seed0_control_seed=101,**args)
        except StrictValidationError as exc:
            assert exc.code=='argument.control_seed';rejected.append(dict(name='seed_exception_on_audited_cell',code=exc.code))
        else:raise AssertionError('Audited seed exception accepted')

        # Tape hashes and cardinality are checked before ledger verification.
        op=Path(case.metadata['oracle_parquet']);saved=op.read_bytes();op.write_bytes(saved+b'tamper')
        try:load_pinned_reference(case)
        except StrictValidationError as exc:
            assert exc.code=='reference.hash';rejected.append(dict(name='pinned_tape_bytes_changed',code=exc.code))
        else:raise AssertionError('Changed pinned bytes accepted')
        op.write_bytes(saved)
        duplicate=pd.concat([oracle_frame,oracle_frame.iloc[:1]],ignore_index=True);duplicate.to_parquet(op,index=False)
        case.metadata['source_hashes'][op.name]=sha(op)
        try:load_pinned_reference(case)
        except StrictValidationError as exc:
            assert exc.code=='reference.cardinality';rejected.append(dict(name='duplicate_original_tape_key',code=exc.code))
        else:raise AssertionError('Duplicate tape row accepted')
        op.write_bytes(saved);case.metadata['source_hashes'][op.name]=sha(op)
        hp=Path(case.metadata['helper_parquet']);helper_saved=hp.read_bytes()
        helper_frame.iloc[1:].to_parquet(hp,index=False);case.metadata['source_hashes'][hp.name]=sha(hp)
        try:load_pinned_reference(case)
        except StrictValidationError as exc:
            assert exc.code=='reference.helper_missing';rejected.append(dict(name='missing_original_helper_key',code=exc.code))
        else:raise AssertionError('Missing helper row accepted')
        hp.write_bytes(helper_saved);case.metadata['source_hashes'][hp.name]=sha(hp)
        # Exercise real historical result shapes on our own fresh tiny tapes:
        # exact flat summaries/ledger alias/affine cover, and direct/count/cut
        # audited and point result conventions. No historical raw is loaded.
        from exact_baselines import METHODS, run_exact
        from direct_arms import run_direct
        from count_allocation_arms import run_count_allocation
        from cut_arms import run_cut
        from extra_exact_controls import STRATEGIES, run_extra_exact
        integration=[]
        jobs=[(m,0,0,lambda method=m:run_exact(case,method),None) for m in METHODS]
        jobs += [(m,0,0,lambda method=m:run_extra_exact(case,method),None) for m in STRATEGIES]
        jobs += [
            ('direct_feedback_audit_degree',101,1,lambda:run_direct(case,101,1,helper=False),None),
            ('count_allocation_feedback_audit_degree',101,1,lambda:run_count_allocation(case,101,1),None),
            ('cut_feedback_audit',101,1,lambda:run_cut(case,101,1,audit=True,feedback=True),None),
            ('direct_feedback_point_degree',0,0,lambda:run_direct(case,101,0,audit=False,helper=False),101),
            ('count_allocation_feedback_point_degree',0,0,lambda:run_count_allocation(case,101,0,audit=False),101),
            ('cut_feedback_point',0,0,lambda:run_cut(case,101,0,audit=False,feedback=True),101),
        ]
        for method,seed,budget,run,control_seed in jobs:
            actual=copy.deepcopy(record);actual_result=run();actual['result']=actual_result
            actual.update(method=method,seed=seed,budget=budget,key=f'{case.case_id}__{method}__s{seed}__b{budget}')
            options={**args,'expected_method':method,'expected_seed':seed,'expected_budget':budget,
                     'expected_key':actual['key'],'record_path':HERE/(actual['key']+'.json.gz'),
                     'seed0_control_seed':control_seed}
            try:strict_check_record(actual,case,reference=reference,**options)
            except Exception as exc:raise AssertionError('Original result integration: '+method) from exc
            integration.append(method)
        positives.append(dict(name='actual_original_algorithm_result_shapes',methods=integration,executions=len(integration)))
        case.close_sql()

        for nside,empty in ((0,True),(4998,False)):
            target,_,_=fixture(folder,nside,empty=empty,name='strict_size_'+str(nside))
            native_ref=load_pinned_reference(target)
            exact,e_args=make_record(target,exact=True)
            checked=strict_check_record(exact,target,reference=native_ref,**e_args)
            positives.append(dict(name='empty_target' if empty else '9996_repeated_row_ids_two_predicates',identities=target.n,
                                  oracle_checks=checked['pinned_oracle_charge_checks'],helper_checks=checked['pinned_helper_charge_checks']))
            if not empty:
                assert len(set(target.row_ids))==4998 and len(set(native_ref.row_predicates))==9996
                assert native_ref.oracle_tokens_in[0]!=native_ref.oracle_tokens_in[4998]
            target.close_sql()

    # Read-only original-tape JOIN smoke test, not a historical-policy run.
    from native_discussion_cases import load_native_cases
    native_cases=load_native_cases();native=native_cases[0]
    native_ref=load_pinned_reference(native)
    assert native.n==9996 and len(native_ref.labels)==9996 and len(set(native.row_ids))==4998
    native_join=dict(case_id=native.case_id,identities=native.n,unique_row_ids=len(set(native.row_ids)),
                     unique_row_predicate_pairs=len(set(native_ref.row_predicates)),source_hashes=dict(native_ref.source_hashes))
    for c in native_cases:c.close_sql()
    result=dict(status='pass',author_work=True,scope='New verifier tests, temporary synthetic Parquet and one native original-tape join only; no full historical verification',
                positive_cases=positives,rejected_mutations=rejected,native_original_tape_join=native_join,
                strict_module_sha256=sha(HERE/'strict_result_validation.py'),checks_source_sha256=sha(Path(__file__)),
                elapsed_seconds=time.perf_counter()-started)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
