"""CPU-only wiring check using historical labels; never live-cost evidence."""
import json
from pathlib import Path
import time

import native_live as driver


def main():
    driver.os.chdir(driver.RELEASE)
    cases = driver.load_cases(5000)
    mapping = {}
    for case in cases:
        truth = driver.common.Oracle(case)._validation_labels()
        for j in range(case.n):
            predicate = case.predicate_left if case.side[j] == 0 else case.predicate_right
            p = driver.prompt(case.texts[j], predicate.rsplit('/',1)[-1])
            y = int(truth[j])
            if p in mapping:
                assert mapping[p] == y
            mapping[p] = y

    class TapeClient:
        def infer(self, prompts):
            return dict(wall_seconds=0., responses=[dict(label=mapping[p], tokens_in=0,
                tokens_out=0, timings={}, raw=dict(test_only=True)) for p in prompts])

    out = driver.RELEASE/'outputs/native_wiring_test'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in driver.METHODS:
        record = driver.execute(TapeClient(), cases[0], method, 101, 500, out)
        rows.append(dict(method=method, requests=record['row']['requests'],
            bag_subset=record['result']['output']['bag_subset'],
            helper_requests=record['row']['helper_requests'],
            cold_returned_first=record['result']['output']['cold_returned_first']))
    result = dict(status='pass', role='CPU wiring validation only; not live-cost evidence',
        historical_labels=True, gpu_inference=False, rows=rows)
    driver.study.save(out/'summary.json', result)
    print(json.dumps(result), flush=True)
    for case in cases:
        case.close_sql()


if __name__ == '__main__':
    main()
