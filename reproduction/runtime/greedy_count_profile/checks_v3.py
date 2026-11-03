"""Regression for valid evidence-conflict attempts before planner entry."""
from pathlib import Path
import hashlib
import json
import sys

from policy_adapter_v3 import run_greedy_count
from policy_adapter_v2 import _execution
from common import make_tiny_case

HERE = Path(__file__).resolve().parent


def main():
    original_regions = _execution.make_regions
    try:
        _execution.make_regions = lambda *args: ({(0, 1): (1, 1), (2, 3): (1, 1)}, [], 0.)
        case = make_tiny_case(side=[0, 0, 1, 1], key_left=[(0, 1)], key_right=[(2, 3)],
                              truth=[1, 1, 1, 1], name='preplanner_conflict')
        result = run_greedy_count(case, 101, 0, audit=False)
        assert result['conflict_freeze'] is True
        assert result['final_scaled_support'] is None
        assert result['projection_calls'] == 2
        assert result['planner_invocations'] == len(result['planner_observations']) == 1
        assert result['projection_attempts_without_planner'] == 1
        assert result['initial_discarded'] == result['discarded']
        assert result['events'][-1]['chosen_counts'] is None
        assert result['output']['full_count'] == result['output']['returned_count'] == 4
        actual = case._sql['connection'].execute('SELECT key_id,left_id,right_id FROM returned_derivations ORDER BY ALL').fetchall()
        assert actual == [(0, 0, 2), (0, 0, 3), (0, 1, 2), (0, 1, 3)]
        case.close_sql()
    finally:
        _execution.make_regions = original_regions
    # A covered ordinary policy has no missing planner entries.
    case = make_tiny_case(side=[0, 0, 1, 1], key_left=[(0, 1)], key_right=[(2, 3)],
                          truth=[1, 0, 0, 1], name='reached_planner')
    normal = run_greedy_count(case, 101, 1)
    assert normal['projection_calls'] == normal['planner_invocations']
    assert normal['projection_attempts_without_planner'] == 0
    case.close_sql()
    out = dict(status='pass', preplanner_evidence_conflict_preserved=True, conflict_projection_attempts=2,
               conflict_planner_invocations=1, conflict_unentered_attempts=1, final_none_support_preserved=True,
               actual_sql_conflict_case_exact=True, normal_entered_accounting_checked=True,
               source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (HERE/'policy_adapter_v3.py', Path(__file__))})
    destination = Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'checks_v3_result.json'
    destination.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out), flush=True)


if __name__ == '__main__': main()
