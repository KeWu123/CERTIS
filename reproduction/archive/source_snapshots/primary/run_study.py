"""Resumable complete matrix, with immutable source snapshots and raw ledgers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import gzip
import hashlib
import json
import platform
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from common import load_cases
from exact_baselines import run_exact, METHODS
from direct_arms import run_direct


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Fraction): return str(value)
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Unserializable {type(value).__name__}")


def sources(output):
    paths = [HERE / p for p in ("common.py", "exact_baselines.py", "direct_arms.py", "cut_arms.py", "run_study.py")]
    paths += [ROOT / "code" / p for p in ("bilinear_certificate.py", "relational_cut.py", "exact_hypergeom.py")]
    paths += [HERE.parent / "EXTERNAL_PROTOCOL.md", HERE.parent / "external_inputs/validation.json"]
    records = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}
    source_id = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    frozen = output / "sources" / source_id
    frozen.mkdir(parents=True, exist_ok=True)
    for p in paths:
        target = frozen / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(p.read_bytes())
        if digest(target) != records[str(p.relative_to(ROOT)).replace("\\", "/")]:
            raise AssertionError("Frozen source changed")
    (frozen / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return source_id, records


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, default=convert, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(gzip.compress(body, compresslevel=5, mtime=0))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", default=str(HERE / "results"))
    args = parser.parse_args()
    from cut_arms import run_cut
    output = Path(args.output)
    source_id, manifest = sources(output)
    directory = output / source_id
    directory.mkdir(parents=True, exist_ok=True)
    machine = {"platform": platform.platform(), "python": platform.python_version(),
               "numpy": np.__version__, "processor": platform.processor(),
               "live_inference": False, "sql_threads": 1}
    cells = [(101, 125)] if args.phase == "pilot" else [(101, 125), (202, 125), (303, 125), (101, 500), (202, 500), (303, 500)]
    cases = load_cases()
    journal = directory / "journal.jsonl"
    finished = 0; failures = 0; skipped = 0
    for case in cases:
        jobs = [(method, 0, 0, lambda m=method: run_exact(case, m)) for method in METHODS]
        jobs += [("direct_feedback_point_helper", 0, 0,
                  lambda: run_direct(case, 101, 0, audit=False)),
                 ("cut_feedback_point", 0, 0,
                  lambda: run_cut(case, 101, 0, audit=False, feedback=True))]
        for seed, budget in cells:
            jobs += [("direct_feedback_audit_helper", seed, budget,
                      lambda s=seed, b=budget: run_direct(case, s, b)),
                     ("direct_static_audit_helper", seed, budget,
                      lambda s=seed, b=budget: run_direct(case, s, b, feedback=False)),
                     ("direct_feedback_audit_degree", seed, budget,
                      lambda s=seed, b=budget: run_direct(case, s, b, helper=False)),
                     ("keywise_feedback_audit_helper", seed, budget,
                      lambda s=seed, b=budget: run_direct(case, s, b, keywise=True)),
                     ("cut_feedback_audit", seed, budget,
                      lambda s=seed, b=budget: run_cut(case, s, b, audit=True, feedback=True)),
                     ("cut_static_audit", seed, budget,
                      lambda s=seed, b=budget: run_cut(case, s, b, audit=True, feedback=False))]
        for name, seed, budget, work in jobs:
            key = f"{case.case_id}__{name}__s{seed}__b{budget}"
            path = directory / "raw" / f"{key}.json.gz"
            if path.exists():
                old = json.loads(gzip.decompress(path.read_bytes()))
                if old["status"] == "pass" and old["source_id"] == source_id:
                    skipped += 1; continue
                # Failed runs remain immutable and visible in the journal.
                path = directory / "raw" / f"{key}__retry{time.time_ns()}.json.gz"
            started = time.perf_counter()
            record = {"key": key, "case_id": case.case_id, "dataset": case.dataset,
                      "predicate_left": case.predicate_left, "predicate_right": case.predicate_right,
                      "variant": case.variant, "method": name, "seed": seed, "budget": budget,
                      "source_id": source_id, "machine": machine,
                      "target_version": case.target_version, "oracle_version": case.oracle_version,
                      "n": case.n, "key_count": len(case.key_names), "metadata": case.metadata}
            try:
                result = work()
                if result.get("summary") is None:
                    result["summary"] = {k: result[k] for k in (
                        "oracle_requests", "oracle_tokens_in", "oracle_tokens_out", "helper_requests",
                        "helper_tokens_in", "helper_tokens_out", "phase_counts", "phase_tokens")}
                record.update(status="pass", result=result)
                if not result["output"]["bag_subset"]:
                    raise AssertionError("Returned bag is not a subset")
                # Record true loss violations rather than deleting them. Their
                # existence is not a runtime error or proof that E occurred.
                out = result["output"]
                record["observed_loss_target_met"] = 10*out["lost_count"] <= out["full_count"]
                finished += 1
            except Exception as exc:
                record.update(status="failure", error=repr(exc), traceback=traceback.format_exc())
                failures += 1
            record["wall_seconds_including_verification"] = time.perf_counter() - started
            save(path, record)
            event = {"key": key, "status": record["status"], "seconds": record["wall_seconds_including_verification"], "path": str(path)}
            with journal.open("a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        case.close_sql()
    end = {"phase": args.phase, "source_id": source_id, "executed_pass": finished,
           "executed_fail": failures, "reused_verified_runs": skipped,
           "expected_unique_runs": len(cases)*(7+6*len(cells)), "source_manifest": manifest}
    (directory / f"{args.phase}_completion.json").write_text(json.dumps(end, indent=2), encoding="utf-8")
    print(json.dumps(end), flush=True)
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
