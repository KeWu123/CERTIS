"""Measure returned-bag SQL before constructing an independent reference.

This evaluator is for the native live protocol only.  Its standard output
fields match ``common.evaluate``; production SQL costs remain the sum of
``sql_preprocessing_seconds``, ``fact_ingest_seconds``, and
``returned_materialization_seconds``.  Reference construction, count checks,
and bag validation are separate harness costs.  No reference fact is inserted
into ``acquired_facts``.

Import after the release's external-study directory is on ``sys.path``.  Use
this evaluator for every policy, with ``case.close_sql()`` before each fresh
policy run.  Existing ``Case.close_sql`` handles its connection state.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np
import pandas as pd

import common


RETURNED_SQL = """CREATE OR REPLACE TEMP TABLE returned_derivations AS
SELECT l.key_id,l.global_id AS left_id,r.global_id AS right_id
FROM incidence l JOIN incidence r ON l.key_id=r.key_id AND l.side=0 AND r.side=1
JOIN acquired_facts pl ON pl.global_id=l.global_id AND pl.label=1
JOIN acquired_facts pr ON pr.global_id=r.global_id AND pr.label=1"""

REFERENCE_SQL = """CREATE TEMP TABLE full_derivations AS
SELECT l.key_id,l.global_id AS left_id,r.global_id AS right_id
FROM incidence l JOIN incidence r ON l.key_id=r.key_id AND l.side=0 AND r.side=1
JOIN reference_labels pl ON pl.global_id=l.global_id AND pl.label=1
JOIN reference_labels pr ON pr.global_id=r.global_id AND pr.label=1"""


def _production_state(case: common.Case) -> dict:
    """Build only known metadata tables; never inspect an oracle reference."""
    started = time.perf_counter()
    connection = common.duckdb.connect()
    try:
        connection.execute("SET threads=1")
        incidences = [
            (k, j, side)
            for k, pair in enumerate(zip(case.key_left, case.key_right))
            for side, ids in enumerate(pair)
            for j in ids
        ]
        frame = pd.DataFrame(incidences, columns=("key_id", "global_id", "side"))
        connection.register("incidence_input", frame)
        connection.execute(
            "CREATE TEMP TABLE incidence AS SELECT key_id::BIGINT key_id,"
            "global_id::BIGINT global_id,side::TINYINT side FROM incidence_input"
        )
        connection.unregister("incidence_input")
        nodes = pd.DataFrame(
            {
                "global_id": np.arange(case.n, dtype=np.int64),
                "row_id": case.row_ids,
                "predicate_id": [
                    case.predicate_left if case.side[j] == 0 else case.predicate_right
                    for j in range(case.n)
                ],
            }
        )
        connection.register("nodes_input", nodes)
        connection.execute("CREATE TEMP TABLE nodes AS SELECT * FROM nodes_input")
        connection.unregister("nodes_input")
    except BaseException:
        connection.close()
        raise
    return {
        "connection": connection,
        "preprocessing_seconds": time.perf_counter() - started,
        "returned_first_evaluator": True,
        "reference_ready": False,
    }


def _reference_state(case: common.Case, state: dict) -> None:
    """Construct harness-only reference tables after returned SQL has run."""
    connection = state["connection"]
    started = time.perf_counter()
    if "oracle_parquet" in case.metadata:
        connection.execute(
            "CREATE TEMP TABLE reference_labels AS SELECT n.global_id,o.label "
            "FROM nodes n JOIN read_parquet(?) o ON n.row_id=o.row_id "
            "AND n.predicate_id=o.predicate_id",
            [str(case.metadata["oracle_parquet"])],
        )
    else:
        # The array fallback is for tiny fixtures, as in the release harness.
        labels = common.Oracle(case)._validation_labels()
        frame = pd.DataFrame(
            {"global_id": np.arange(case.n, dtype=np.int64), "label": labels}
        )
        connection.register("reference_input", frame)
        connection.execute("CREATE TEMP TABLE reference_labels AS SELECT * FROM reference_input")
        connection.unregister("reference_input")
    state["reference_label_loading_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    connection.execute(REFERENCE_SQL)
    state["reference_materialization_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    state["full_count"] = int(connection.execute("SELECT count(*) FROM full_derivations").fetchone()[0])
    state["reference_count_seconds"] = time.perf_counter() - started
    state["reference_sql"] = REFERENCE_SQL
    state["reference_ready"] = True


def returned_first_evaluate(case: common.Case, facts: Mapping[int, int]) -> dict:
    """Materialize paid facts first, then validate against the fixed oracle.

    A cached SQL state reuses incidence/nodes and the full reference, just as
    ``common.evaluate`` does.  Their one-time costs are then reported as zero;
    facts ingestion, returned CTAS, returned counts, and EXCEPT ALL checks run
    again.  Fresh policy measurements must call ``case.close_sql()`` first.
    """
    acquired = common._facts(facts, case.n)
    cached = case._sql is not None
    if not cached:
        case._sql = _production_state(case)
    state = case._sql
    connection = state["connection"]
    try:
        started = time.perf_counter()
        frame = pd.DataFrame(
            {
                "global_id": np.asarray(tuple(acquired), dtype=np.int64),
                "label": np.asarray(tuple(acquired.values()), dtype=np.int8),
            }
        )
        connection.register("facts_input", frame)
        connection.execute("CREATE OR REPLACE TEMP TABLE acquired_facts AS SELECT * FROM facts_input")
        connection.unregister("facts_input")
        fact_ingest_seconds = time.perf_counter() - started

        started = time.perf_counter()
        connection.execute(RETURNED_SQL)
        returned_materialization_seconds = time.perf_counter() - started

        # Reference access starts only after the production CTAS above.  The
        # comparison cannot populate the policy cache or rescue an unpaid row.
        started = time.perf_counter()
        labels = common.Oracle(case)._validation_labels()
        if any(int(labels[j]) != value for j, value in acquired.items()):
            raise ValueError("Returned facts contradict the immutable oracle tape")
        acquired_check_seconds = time.perf_counter() - started

        # A state from common.evaluate already has a complete reference.  It
        # can be reused for non-cold diagnostic calls without a second CTAS.
        reference_cached = bool(state.get("reference_ready", "full_count" in state))
        if not reference_cached:
            _reference_state(case, state)

        started = time.perf_counter()
        inconsistent = int(
            connection.execute(
                "SELECT count(*) FROM acquired_facts f LEFT JOIN reference_labels r "
                "ON f.global_id=r.global_id "
                "WHERE r.global_id IS NULL OR f.label<>r.label"
            ).fetchone()[0]
        )
        if inconsistent:
            raise ValueError("Acquired facts contradict the independent SQL oracle reference")
        returned_count = int(connection.execute("SELECT count(*) FROM returned_derivations").fetchone()[0])
        extra = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT * FROM returned_derivations "
                "EXCEPT ALL SELECT * FROM full_derivations)"
            ).fetchone()[0]
        )
        missing = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT * FROM full_derivations "
                "EXCEPT ALL SELECT * FROM returned_derivations)"
            ).fetchone()[0]
        )
        full_count = state["full_count"]
        if extra or missing != full_count - returned_count:
            raise AssertionError("Materialized bag subset/cardinality validation failed")
        validation_seconds = time.perf_counter() - started
    except BaseException:
        case.close_sql()
        raise

    return {
        "full_count": full_count,
        "returned_count": returned_count,
        "lost_count": missing,
        "loss_numerator": missing,
        "loss_denominator": full_count,
        "loss": missing / full_count if full_count else 0.0,
        "bag_subset": extra == 0,
        "exact": missing == 0,
        "materialized_table": "returned_derivations",
        "full_reference_cached": cached,
        "reference_sql": state["reference_sql"],
        "returned_sql": RETURNED_SQL,
        "sql_preprocessing_seconds": 0.0 if cached else state["preprocessing_seconds"],
        "reference_materialization_seconds": (
            0.0 if reference_cached else state["reference_materialization_seconds"]
        ),
        "reference_label_loading_seconds": (
            0.0 if reference_cached else state["reference_label_loading_seconds"]
        ),
        "fact_ingest_seconds": fact_ingest_seconds,
        "returned_materialization_seconds": returned_materialization_seconds,
        "sql_validation_seconds": validation_seconds,
        "source_hashes": case.metadata.get("source_hashes", {}),
        "target_version": case.target_version,
        "oracle_version": case.oracle_version,
        # Additional diagnostics do not change the standard charged-time sum.
        "reference_count_seconds": 0.0 if reference_cached else state["reference_count_seconds"],
        "acquired_facts_reference_check_seconds": acquired_check_seconds,
        "sql_evaluation_order": "returned-before-reference" if not cached else "cached-reference",
        "cold_returned_first": bool(not cached),
        "sql_validation_includes_returned_count": True,
    }
