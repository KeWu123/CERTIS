"""Small CPU-only checks for the native live returned-first SQL evaluator."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORKSPACE = Path(__file__).resolve().parents[2]
RELEASE = WORKSPACE
sys.path.insert(0, str(RELEASE / "reproduction/runtime/external_study"))

import common
import pandas as pd
from cost_sql import returned_first_evaluate


class ReturnedFirstTests(unittest.TestCase):
    def setUp(self):
        self.cases = []

    def tearDown(self):
        for case in self.cases:
            case.close_sql()

    def case(self, *, multiple=False, zero=False):
        if multiple:
            case = common.make_tiny_case(
                side=[0, 0, 1, 1], key_left=[(0, 1), (0,)],
                key_right=[(2, 3), (2,)], truth=[1, 1, 1, 1],
                name="multiplicity",
            )
        else:
            case = common.make_tiny_case(
                side=[0, 0, 0, 1, 1, 1], key_left=[(0, 1, 2)],
                key_right=[(3, 4, 5)],
                truth=[0] * 6 if zero else [1, 1, 0, 1, 0, 1],
                name="single",
            )
        self.cases.append(case)
        return case

    def test_exact_subset_and_empty_match_release_counts(self):
        for facts in ({0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}, {0: 1, 3: 1}, {}):
            with self.subTest(facts=facts):
                case = self.case()
                expected = common.evaluate(case, facts)
                case.close_sql()
                observed = returned_first_evaluate(case, facts)
                self.assertTrue(set(expected) <= set(observed))
                for field in (
                    "full_count", "returned_count", "lost_count", "loss_numerator",
                    "loss_denominator", "loss", "bag_subset", "exact",
                    "materialized_table", "full_reference_cached", "reference_sql",
                    "returned_sql", "target_version", "oracle_version", "source_hashes",
                ):
                    self.assertEqual(observed[field], expected[field], field)
                self.assertEqual(observed["sql_evaluation_order"], "returned-before-reference")
                self.assertTrue(observed["cold_returned_first"])

    def test_zero_reference_bag(self):
        observed = returned_first_evaluate(self.case(zero=True), {})
        self.assertEqual((observed["full_count"], observed["returned_count"], observed["loss"]), (0, 0, 0))
        self.assertTrue(observed["exact"])

    def test_shared_pair_retains_two_key_derivations(self):
        case = self.case(multiple=True)
        observed = returned_first_evaluate(case, {0: 1, 2: 1})
        self.assertEqual((observed["full_count"], observed["returned_count"], observed["lost_count"]), (5, 2, 3))
        rows = case._sql["connection"].execute(
            "SELECT key_id,left_id,right_id FROM returned_derivations ORDER BY key_id"
        ).fetchall()
        self.assertEqual(rows, [(0, 0, 2), (1, 0, 2)])

    def test_cache_reuses_reference_but_recomputes_returned_counts(self):
        case = self.case(multiple=True)
        cold = returned_first_evaluate(case, {0: 1, 2: 1})
        connection = case._sql["connection"]
        warm = returned_first_evaluate(case, {0: 1, 1: 1, 2: 1, 3: 1})
        self.assertIs(case._sql["connection"], connection)
        self.assertFalse(cold["full_reference_cached"])
        self.assertTrue(warm["full_reference_cached"])
        self.assertEqual(warm["returned_count"], 5)
        for field in ("sql_preprocessing_seconds", "reference_materialization_seconds", "reference_label_loading_seconds", "reference_count_seconds"):
            self.assertGreater(cold[field], 0, field)
            self.assertEqual(warm[field], 0, field)
        for field in ("fact_ingest_seconds", "returned_materialization_seconds", "sql_validation_seconds"):
            self.assertGreater(warm[field], 0, field)
        self.assertTrue(warm["sql_validation_includes_returned_count"])
        case.close_sql()
        self.assertIsNone(case._sql)
        rebuilt = returned_first_evaluate(case, {})
        self.assertFalse(rebuilt["full_reference_cached"])
        self.assertGreater(rebuilt["sql_preprocessing_seconds"], 0)
        self.assertEqual(rebuilt["returned_count"], 0)

    def test_reference_access_happens_after_returned_ctas(self):
        case = self.case()
        original = common.Oracle._validation_labels
        checks = []

        def check_order(oracle):
            count = oracle.case._sql["connection"].execute(
                "SELECT count(*) FROM returned_derivations"
            ).fetchone()[0]
            checks.append(count)
            return original(oracle)

        with patch.object(common.Oracle, "_validation_labels", check_order), patch.object(
            common, "_setup_sql", side_effect=AssertionError("Old reference-first setup called")
        ):
            observed = returned_first_evaluate(case, {0: 1, 3: 1})
        self.assertEqual(observed["returned_count"], 1)
        self.assertEqual(checks, [1, 1])

    def test_independent_parquet_reference_and_paid_facts(self):
        case = self.case()
        labels = common.Oracle(case)._validation_labels()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "oracle.parquet"
            pd.DataFrame(
                {
                    "row_id": case.row_ids,
                    "predicate_id": [case.predicate_left if side == 0 else case.predicate_right for side in case.side],
                    "label": labels,
                }
            ).to_parquet(path, index=False)
            case.metadata["oracle_parquet"] = str(path)
            facts = {0: 1, 3: 1}
            observed = returned_first_evaluate(case, facts)
            self.assertEqual(facts, {0: 1, 3: 1})
            self.assertEqual(observed["full_count"], 4)
            self.assertEqual(observed["returned_count"], 1)
            acquired = case._sql["connection"].execute(
                "SELECT * FROM acquired_facts ORDER BY global_id"
            ).fetchall()
            self.assertEqual(acquired, [(0, 1), (3, 1)])
            case.close_sql()

    def test_contradictory_paid_fact_is_rejected_and_connection_closed(self):
        case = self.case()
        with self.assertRaisesRegex(ValueError, "contradict"):
            returned_first_evaluate(case, {2: 1})
        self.assertIsNone(case._sql)

    def test_cache_from_common_evaluator_remains_compatible(self):
        case = self.case()
        common.evaluate(case, {0: 1, 3: 1})
        observed = returned_first_evaluate(case, {})
        self.assertTrue(observed["full_reference_cached"])
        self.assertEqual(observed["sql_preprocessing_seconds"], 0)
        self.assertEqual(observed["reference_materialization_seconds"], 0)
        self.assertEqual(observed["returned_count"], 0)
        self.assertEqual(observed["sql_evaluation_order"], "cached-reference")
        self.assertFalse(observed["cold_returned_first"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
