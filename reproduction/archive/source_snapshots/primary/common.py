"""Locked external-study metadata, charged tape access and DuckDB validation.

Algorithms receive Case metadata and Oracle facts; truth/token arrays remain
in this harness module. Python access control is a convention, not a sandbox.
Scores used by an algorithm must pass through Oracle.use_scores for charging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any
import hashlib
import importlib.util
import json
import sys
import time

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WORKSPACE = STUDY.parents[1]
SOURCE = STUDY / "candidate_systems/external/semviews"
INPUTS = STUDY / "external_inputs"
PHASES = ("audit", "cut", "execution")
DBPEDIA_PAIRS = (("agent", "person"), ("person", "athlete"),
    ("organization", "company"), ("place", "natural_place"),
    ("work", "musical_work"), ("event", "societal_event"),
    ("athlete", "sports_team"), ("company", "educational_institution"))
GOEMOTIONS_PAIRS = (("sent_positive", "joy"), ("sent_negative", "anger"),
    ("sent_ambiguous", "confusion"), ("is_question", "curiosity"),
    ("mentions_person", "admiration"), ("sent_negative", "sadness"),
    ("sarcastic", "annoyance"), ("sent_positive", "love"))


@dataclass(frozen=True)
class _Tape:
    labels: np.ndarray
    oracle_input_tokens: np.ndarray
    oracle_output_tokens: np.ndarray
    helper_input_tokens: np.ndarray
    helper_output_tokens: np.ndarray


_TAPES: dict[str, _Tape] = {}
_CUT_MODULE = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids(values: Iterable[int], n: int) -> tuple[int, ...]:
    result = []
    seen = set()
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or not 0 <= int(v) < n:
            raise ValueError("In-target integer identity required")
        j = int(v)
        if j not in seen:
            result.append(j); seen.add(j)
    return tuple(result)


def _facts(facts: Mapping[int, int], n: int) -> dict[int, int]:
    if not isinstance(facts, Mapping):
        raise ValueError("Facts must be a mapping")
    out = {}
    for j, y in facts.items():
        idx = _ids((j,), n)[0]
        if not isinstance(y, Integral) or int(y) not in (0, 1):
            raise ValueError("Binary integer point facts required")
        out[idx] = int(y)
    return out


@dataclass(eq=False)
class Case:
    dataset: str
    predicate_left: str
    predicate_right: str
    variant: str
    ids: tuple[str, ...]
    side: np.ndarray
    key_names: tuple[Any, ...]
    key_left: tuple[tuple[int, ...], ...]
    key_right: tuple[tuple[int, ...], ...]
    scores: np.ndarray
    row_ids: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    row_metadata: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)
    _tape_key: str = field(default="", repr=False)
    _sql: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.ids = tuple(self.ids)
        self.side = np.asarray(self.side, dtype=np.int8).copy()
        self.scores = np.asarray(self.scores, dtype=np.float64).copy()
        self.key_names = tuple(self.key_names)
        self.key_left = tuple(tuple(int(v) for v in a) for a in self.key_left)
        self.key_right = tuple(tuple(int(v) for v in a) for a in self.key_right)
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Left/right oracle identities must be disjoint and unique")
        if self.side.shape != (self.n,) or self.scores.shape != (self.n,):
            raise ValueError("One side/score per identity required")
        if np.any((self.side != 0) & (self.side != 1)) or np.any(np.diff(self.side) < 0):
            raise ValueError("Identities must be ordered by side then row_id")
        if not np.all(np.isfinite(self.scores)) or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("Finite helper scores in [0,1] required")
        if len(self.key_names) != len(set(self.key_names)) or len(self.key_left) != len(self.key_names) or len(self.key_right) != len(self.key_names):
            raise ValueError("One unique named key with both incidence groups required")
        for a, b in zip(self.key_left, self.key_right):
            if a != tuple(sorted(set(a))) or b != tuple(sorted(set(b))):
                raise ValueError("Distinct sorted incidences required within every side/key")
            if any(not 0 <= j < self.n or self.side[j] != 0 for j in a) or any(not 0 <= j < self.n or self.side[j] != 1 for j in b):
                raise ValueError("Wrong-side or out-of-target incidence")
            if set(a) & set(b):
                raise ValueError("Left/right self-alias rejected")
        if self.row_ids and len(self.row_ids) != self.n:
            raise ValueError("One released row_id per identity required")
        for values in (self.side, self.scores):
            values.setflags(write=False)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def case_id(self) -> str:
        return f"{self.dataset}__{self.predicate_left.rsplit('/',1)[-1]}__{self.predicate_right.rsplit('/',1)[-1]}__{self.variant}"

    @property
    def target_version(self) -> str:
        return str(self.metadata.get("target_version", self.case_id))

    @property
    def oracle_version(self) -> str:
        return str(self.metadata.get("oracle_contract_version", "synthetic-fixed-tape"))

    def full_sql_and_materialize(self, facts: Mapping[int, int]) -> dict:
        return evaluate(self, facts)

    def close_sql(self) -> None:
        if self._sql is not None:
            self._sql["connection"].close()
            self._sql = None


class Oracle:
    """Per-policy private cache; only this class reads truth/token tape arrays."""
    def __init__(self, case: Case, initial_facts: Mapping[int, int] | None = None):
        if case._tape_key not in _TAPES:
            raise ValueError("Case has no registered harness tape")
        self.case = case
        self.__tape = _TAPES[case._tape_key]
        self.facts = _facts(initial_facts or {}, case.n)
        if any(self.__tape.labels[j] != y for j, y in self.facts.items()):
            raise ValueError("Initial cache contradicts the pinned immutable tape")
        self.__initial = dict(self.facts)
        self.ledger: list[dict] = []
        self.helper_ledger: list[dict] = []
        self.__helper_ids = set()
        self.helper_read_calls = 0
        self.helper_read_items = 0

    def acquire(self, ids: Iterable[int], phase: str) -> dict[int, int]:
        if phase not in PHASES:
            raise ValueError("phase must be audit/cut/execution")
        new = {}
        for j in _ids(ids, self.case.n):
            if j in self.facts:
                continue
            y = int(self.__tape.labels[j])
            new[j] = y; self.facts[j] = y
            self.ledger.append({"id": j, "identity": self.case.ids[j], "phase": phase, "label": y,
                "tokens_in": int(self.__tape.oracle_input_tokens[j]),
                "tokens_out": int(self.__tape.oracle_output_tokens[j])})
        return new

    def use_scores(self, ids: Iterable[int]) -> np.ndarray:
        # Preserve caller order and repeated reads in returned values, while
        # charging each helper identity/version once per policy.
        values = tuple(ids)
        checked = tuple(_ids((j,), self.case.n)[0] for j in values)
        self.helper_read_calls += 1; self.helper_read_items += len(checked)
        for j in checked:
            if j not in self.__helper_ids:
                self.__helper_ids.add(j)
                self.helper_ledger.append({"id": j, "identity": self.case.ids[j],
                    "tokens_in": int(self.__tape.helper_input_tokens[j]),
                    "tokens_out": int(self.__tape.helper_output_tokens[j])})
        return self.case.scores[np.asarray(checked, dtype=np.int64)].copy()

    def summary(self) -> dict:
        paid = {r["id"]: r["label"] for r in self.ledger}
        if self.facts != {**self.__initial, **paid}:
            raise ValueError("Facts were mutated outside charged Oracle acquisition")
        result = {"oracle_requests": len(self.ledger), "initial_cache_count": len(self.__initial),
            "oracle_tokens_in": sum(r["tokens_in"] for r in self.ledger),
            "oracle_tokens_out": sum(r["tokens_out"] for r in self.ledger),
            "helper_requests": len(self.helper_ledger),
            "helper_tokens_in": sum(r["tokens_in"] for r in self.helper_ledger),
            "helper_tokens_out": sum(r["tokens_out"] for r in self.helper_ledger),
            "helper_read_calls": self.helper_read_calls, "helper_read_items": self.helper_read_items,
            "phase_counts": {p: sum(r["phase"] == p for r in self.ledger) for p in PHASES},
            "phase_tokens": {p: {"tokens_in": sum(r["tokens_in"] for r in self.ledger if r["phase"] == p),
                                  "tokens_out": sum(r["tokens_out"] for r in self.ledger if r["phase"] == p)} for p in PHASES},
            "helper_scope": "shared pretrained signal; upstream helper-selection/training cost unknown",
            "latency_scope": "recorded tokens only; no live inference latency measurement"}
        return result

    def _validation_labels(self) -> np.ndarray:
        """Harness-only full reference access; never exposed to algorithms."""
        return self.__tape.labels


def _retained(case: Case, retained=None) -> np.ndarray:
    if retained is None:
        return np.ones(case.n, dtype=bool)
    if isinstance(retained, np.ndarray) and retained.dtype == np.bool_:
        if retained.shape != (case.n,):
            raise ValueError("Retained mask shape mismatch")
        return retained.copy()
    mask = np.zeros(case.n, dtype=bool)
    mask[list(_ids(retained, case.n))] = True
    return mask


def potential_degrees(case: Case, facts: Mapping[int, int], retained=None,
                      opposite_scores=None) -> np.ndarray:
    """Bag potential degree against nonnegative possible retained partners.

    A known positive is always rescued. A shared key contributes separately;
    no pair deduplication or hidden oracle label is used.
    """
    f = _facts(facts, case.n); keep = _retained(case, retained)
    alive = keep.copy()
    for j, y in f.items():
        alive[j] = bool(y)
    scores = np.ones(case.n, dtype=np.float64) if opposite_scores is None else np.asarray(opposite_scores, dtype=np.float64)
    if scores.shape != (case.n,) or np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("One finite nonnegative partner score per identity required")
    result = np.zeros(case.n, dtype=np.float64)
    for left, right in zip(case.key_left, case.key_right):
        ls = sum(float(scores[j]) for j in left if alive[j])
        rs = sum(float(scores[j]) for j in right if alive[j])
        for j in left:
            if alive[j]: result[j] += rs
        for j in right:
            if alive[j]: result[j] += ls
    return result


def possible_unknowns(case: Case, facts: Mapping[int, int], retained=None) -> tuple[int, ...]:
    degree = potential_degrees(case, facts, retained)
    return tuple(j for j in range(case.n) if j not in facts and degree[j] > 0)


def load_cut_module():
    global _CUT_MODULE
    if _CUT_MODULE is None:
        name = "external_study_existing_relational_cut"
        spec = importlib.util.spec_from_file_location(name, WORKSPACE / "code/relational_cut.py")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        old = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try: spec.loader.exec_module(module)
        finally: sys.dont_write_bytecode = old
        _CUT_MODULE = module
    return _CUT_MODULE


def make_join(case: Case):
    core = load_cut_module()
    # Zero-multiplicity records retain keyless target identities in the core
    # namespace without manufacturing an actual matching derivation.
    left = [core.Incidence(j, j, k) for k, ids in enumerate(case.key_left) for j in ids]
    right = [core.Incidence(j, j, k) for k, ids in enumerate(case.key_right) for j in ids]
    seen = {x.identity for x in (*left, *right)}
    for j in range(case.n):
        if j not in seen:
            (left if case.side[j] == 0 else right).append(core.Incidence(j, j, ("isolated", j), 0))
    return core.FactorizedBagJoin(left, right, {j: 1 for j in range(case.n)},
        target_version=case.target_version, oracle_version=case.oracle_version)


def _read_metadata(dataset: str):
    con = duckdb.connect()
    try:
        path = SOURCE / f"data/datasets/{dataset}.parquet"
        cols = "row_id,text,l1,l2,l3" if dataset == "dbpedia" else "row_id,text,emotions"
        rows = con.execute(f"SELECT {cols} FROM read_parquet(?) ORDER BY row_id", [str(path)]).fetchall()
        return rows
    finally: con.close()


def load_cases() -> list[Case]:
    validation = json.loads((INPUTS / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((SOURCE / "download_manifest.json").read_text(encoding="utf-8"))
    source_hashes = {a["path"]: a["sha256"] for a in manifest["files"]}
    cases = []
    for dataset, pairs in (("dbpedia", DBPEDIA_PAIRS), ("goemotions", GOEMOTIONS_PAIRS)):
        record = next(a for a in validation["datasets"] if a["dataset"] == dataset)
        array_path = INPUTS / f"{dataset}_arrays.npz"
        if sha256(array_path) != record["array_sha256"]:
            raise ValueError("Validated array hash mismatch")
        for relative in (f"data/datasets/{dataset}.parquet", f"data/tapes/{dataset}__oracle-llama.parquet"):
            if sha256(SOURCE / relative) != source_hashes[relative]:
                raise ValueError("Pinned source parquet hash mismatch")
        rows = _read_metadata(dataset)
        arrays = np.load(array_path, allow_pickle=False)
        row_ids = tuple(str(x) for x in arrays["row_ids"])
        if len(rows) != 5000 or tuple(r[0] for r in rows) != row_ids or len(set(row_ids)) != 5000:
            raise ValueError("Pinned parquet and sorted array identities disagree")
        pred_index = {str(p): i for i, p in enumerate(arrays["predicate_ids"])}
        side = np.array([0] * 2500 + [1] * 2500, dtype=np.int8)
        for left, right in pairs:
            pl, pr = f"{dataset}/{left}", f"{dataset}/{right}"
            if pl not in pred_index or pr not in pred_index:
                raise ValueError("Protocol predicate absent from validated tapes")
            il, ir = pred_index[pl], pred_index[pr]
            def combined(name):
                value = np.concatenate((arrays[name][il, :2500], arrays[name][ir, 2500:])).copy()
                value.setflags(write=False); return value
            tape_key = f"{dataset}|{pl}|{pr}|{record['oracle_contract_version']}"
            _TAPES[tape_key] = _Tape(*(combined(k) for k in ("labels", "oracle_input_tokens", "oracle_output_tokens", "helper_input_tokens", "helper_output_tokens")))
            oracle_ids = tuple(f"{pl if j < 2500 else pr}|{rid}|{record['oracle_contract_version']}" for j, rid in enumerate(row_ids))
            for variant in ("single", "multi"):
                by_key = {}
                metadata = []
                for j, row in enumerate(rows):
                    if dataset == "dbpedia":
                        meta = {"l1": row[2], "l2": row[3], "l3": row[4]}
                        keys = [("l1", row[2])] if variant == "single" else list(zip(("l1", "l2", "l3"), row[2:5]))
                        keys = [key for key in keys if key[1] is not None]
                    else:
                        annotations = json.loads(row[2]) if isinstance(row[2], str) else list(row[2] or [])
                        annotations = sorted(set(str(a) for a in annotations))
                        meta = {"emotions": tuple(annotations)}
                        keys = annotations[:1] if variant == "single" else annotations
                        if not keys and variant == "single": keys = ["__EMPTY_ANNOTATION__"]
                    metadata.append(meta)
                    for key in set(keys):
                        by_key.setdefault(key, ([], []))[int(side[j])].append(j)
                names = tuple(sorted(by_key))
                version_payload = (dataset, pl, pr, variant, names, row_ids, source_hashes[f"data/datasets/{dataset}.parquet"])
                target_version = "sha256:" + hashlib.sha256(repr(version_payload).encode()).hexdigest()
                case = Case(dataset, pl, pr, variant, oracle_ids, side, names,
                    tuple(tuple(by_key[k][0]) for k in names), tuple(tuple(by_key[k][1]) for k in names),
                    combined("scores"), row_ids, tuple(r[1] for r in rows), tuple(metadata),
                    {"target_version": target_version, "oracle_contract_version": record["oracle_contract_version"],
                     "upstream_commit": validation["upstream_commit"], "array_sha256": record["array_sha256"],
                     "source_hashes": {k: v for k, v in source_hashes.items() if dataset in k},
                     "oracle_parquet": str(SOURCE / f"data/tapes/{dataset}__oracle-llama.parquet"),
                     "truth_scope": record["truth_scope"], "construction": "protocol-locked constructed external metadata join"}, tape_key)
                cases.append(case)
        arrays.close()
    if len(cases) != 32:
        raise AssertionError("Locked protocol requires all 32 cases")
    return cases


def make_tiny_case(*, side, key_left, key_right, truth, scores=None,
                   key_names=None, ids=None, name="tiny", tokens=None) -> Case:
    """Harness-check fixture factory; never supplies production algorithms truth."""
    side = np.asarray(side, dtype=np.int8); n = len(side)
    labels = np.asarray(truth, dtype=np.int8).copy()
    if labels.shape != (n,) or np.any((labels != 0) & (labels != 1)):
        raise ValueError("Binary fixture truth required")
    token_map = tokens or {}
    vectors = []
    for key in ("oracle_input_tokens", "oracle_output_tokens", "helper_input_tokens", "helper_output_tokens"):
        vector = np.asarray(token_map.get(key, np.ones(n, dtype=np.int64)), dtype=np.int64).copy()
        if vector.shape != (n,) or np.any(vector < 0): raise ValueError("Invalid fixture tokens")
        vector.setflags(write=False); vectors.append(vector)
    labels.setflags(write=False)
    tape_key = "tiny:" + hashlib.sha256(repr((name, side.tolist(), key_left, key_right, labels.tolist(), len(_TAPES))).encode()).hexdigest()
    _TAPES[tape_key] = _Tape(labels, *vectors)
    ids = tuple(ids or (f"{name}/{int(side[j])}/row{j}" for j in range(n)))
    return Case(name, name + "/left", name + "/right", "tiny", ids, side,
        tuple(key_names or range(len(key_left))), tuple(key_left), tuple(key_right),
        np.asarray(scores if scores is not None else np.full(n, .5), dtype=np.float64),
        tuple(f"row{j}" for j in range(n)), metadata={"target_version": tape_key}, _tape_key=tape_key)


def _setup_sql(case: Case) -> dict:
    started = time.perf_counter(); c = duckdb.connect()
    c.execute("SET threads=1")
    incidences = [(k, j, s) for k, pair in enumerate(zip(case.key_left, case.key_right)) for s, ids in enumerate(pair) for j in ids]
    frame = pd.DataFrame(incidences, columns=("key_id", "global_id", "side"))
    c.register("incidence_input", frame)
    c.execute("CREATE TEMP TABLE incidence AS SELECT key_id::BIGINT key_id,global_id::BIGINT global_id,side::TINYINT side FROM incidence_input")
    c.unregister("incidence_input")
    nodes = pd.DataFrame({"global_id": np.arange(case.n, dtype=np.int64), "row_id": case.row_ids,
        "predicate_id": [case.predicate_left if case.side[j] == 0 else case.predicate_right for j in range(case.n)]})
    c.register("nodes_input", nodes)
    c.execute("CREATE TEMP TABLE nodes AS SELECT * FROM nodes_input")
    c.unregister("nodes_input")
    preprocessing = time.perf_counter() - started
    label_started = time.perf_counter()
    if "oracle_parquet" in case.metadata:
        oracle_path = case.metadata["oracle_parquet"]
        c.execute("CREATE TEMP TABLE reference_labels AS SELECT n.global_id,o.label FROM nodes n JOIN read_parquet(?) o ON n.row_id=o.row_id AND n.predicate_id=o.predicate_id", [oracle_path])
    else:
        # Tiny fixtures only. Production reference labels are re-read in SQL
        # from the independently pinned parquet, never copied from the arrays.
        labels = Oracle(case)._validation_labels()
        label_frame = pd.DataFrame({"global_id": np.arange(case.n, dtype=np.int64), "label": labels})
        c.register("reference_input", label_frame)
        c.execute("CREATE TEMP TABLE reference_labels AS SELECT * FROM reference_input")
        c.unregister("reference_input")
    reference_label_loading_seconds = time.perf_counter() - label_started
    reference_sql = """CREATE TEMP TABLE full_derivations AS
SELECT l.key_id,l.global_id AS left_id,r.global_id AS right_id
FROM incidence l JOIN incidence r ON l.key_id=r.key_id AND l.side=0 AND r.side=1
JOIN reference_labels pl ON pl.global_id=l.global_id AND pl.label=1
JOIN reference_labels pr ON pr.global_id=r.global_id AND pr.label=1"""
    started = time.perf_counter(); c.execute(reference_sql)
    reference_seconds = time.perf_counter() - started
    full_count = int(c.execute("SELECT count(*) FROM full_derivations").fetchone()[0])
    return {"connection": c, "preprocessing_seconds": preprocessing,
            "reference_label_loading_seconds": reference_label_loading_seconds,
            "reference_materialization_seconds": reference_seconds,
            "full_count": full_count, "reference_sql": reference_sql}


def evaluate(case: Case, facts: Mapping[int, int]) -> dict:
    """Actually materialize returned bags, then independently compare with tape SQL."""
    f = _facts(facts, case.n)
    labels = Oracle(case)._validation_labels()
    if any(int(labels[j]) != y for j, y in f.items()):
        raise ValueError("Returned facts contradict the immutable oracle tape")
    cached = case._sql is not None
    if not cached: case._sql = _setup_sql(case)
    state = case._sql; c = state["connection"]
    started = time.perf_counter()
    frame = pd.DataFrame({"global_id": np.asarray(tuple(f), dtype=np.int64), "label": np.asarray(tuple(f.values()), dtype=np.int8)})
    c.register("facts_input", frame)
    c.execute("CREATE OR REPLACE TEMP TABLE acquired_facts AS SELECT * FROM facts_input")
    c.unregister("facts_input")
    fact_ingest_seconds = time.perf_counter() - started
    returned_sql = """CREATE OR REPLACE TEMP TABLE returned_derivations AS
SELECT l.key_id,l.global_id AS left_id,r.global_id AS right_id
FROM incidence l JOIN incidence r ON l.key_id=r.key_id AND l.side=0 AND r.side=1
JOIN acquired_facts pl ON pl.global_id=l.global_id AND pl.label=1
JOIN acquired_facts pr ON pr.global_id=r.global_id AND pr.label=1"""
    started = time.perf_counter(); c.execute(returned_sql)
    materialize_seconds = time.perf_counter() - started
    started = time.perf_counter()
    returned_count = int(c.execute("SELECT count(*) FROM returned_derivations").fetchone()[0])
    extra = int(c.execute("SELECT count(*) FROM (SELECT * FROM returned_derivations EXCEPT ALL SELECT * FROM full_derivations)").fetchone()[0])
    missing = int(c.execute("SELECT count(*) FROM (SELECT * FROM full_derivations EXCEPT ALL SELECT * FROM returned_derivations)").fetchone()[0])
    full_count = state["full_count"]
    if extra or missing != full_count - returned_count:
        raise AssertionError("Materialized bag subset/cardinality validation failed")
    validation_seconds = time.perf_counter() - started
    return {"full_count": full_count, "returned_count": returned_count, "lost_count": missing,
            "loss_numerator": missing, "loss_denominator": full_count,
            "loss": missing / full_count if full_count else 0., "bag_subset": extra == 0,
            "exact": missing == 0, "materialized_table": "returned_derivations",
            "full_reference_cached": cached, "reference_sql": state["reference_sql"], "returned_sql": returned_sql,
            "sql_preprocessing_seconds": 0. if cached else state["preprocessing_seconds"],
            "reference_materialization_seconds": 0. if cached else state["reference_materialization_seconds"],
            "reference_label_loading_seconds": 0. if cached else state["reference_label_loading_seconds"],
            "fact_ingest_seconds": fact_ingest_seconds, "returned_materialization_seconds": materialize_seconds,
            "sql_validation_seconds": validation_seconds,
            "source_hashes": case.metadata.get("source_hashes", {}),
            "target_version": case.target_version, "oracle_version": case.oracle_version}
