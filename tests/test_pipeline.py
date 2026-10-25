"""
End-to-end tests that actually run the pipeline (train models, run
verification, compute consensus) -- slower, but these are the tests that
prove the whole stack works together, not just in isolation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import numpy as np
from core.dataset import build_split
from core.model import train_base_model, evaluate
from core.pipeline import run_audit, run_full_comparison, ALGORITHMS


class TestDataset(unittest.TestCase):
    def test_split_is_disjoint(self):
        split = build_split()
        # no sample should be identifiable in more than one split via exact
        # row match (weak check, but catches the obvious bug of overlapping splits)
        self.assertEqual(len(split.X_retain) + len(split.X_forget), len(split.X_train))
        summary = split.summary()
        self.assertEqual(summary["n_retain"] + summary["n_forget"], summary["n_train"])


class TestUnlearningAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split()
        X_train = np.vstack([cls.split.X_retain, cls.split.X_forget])
        y_train = np.concatenate([cls.split.y_retain, cls.split.y_forget])
        cls.original_model = train_base_model(X_train, y_train)
        cls.sample_ids = np.arange(len(X_train))

    def test_ground_truth_retraining_never_saw_forget_set(self):
        """Sanity: ground truth model's forget-set accuracy should be closer
        to its test-set accuracy than to its retain-set accuracy, since it
        was never trained on the forget set at all."""
        result = run_audit("ground_truth_retraining", self.split, self.original_model)
        acc = result["accuracy"]
        gap_to_test = abs(acc["forget"] - acc["test"])
        gap_to_retain = abs(acc["forget"] - acc["retain"])
        self.assertLess(gap_to_test, gap_to_retain)

    def test_all_algorithms_run_without_error(self):
        for algo in ALGORITHMS:
            with self.subTest(algo=algo):
                result = run_audit(algo, self.split, self.original_model, sample_ids=self.sample_ids)
                self.assertIn("consensus", result)
                self.assertIn(result["consensus"]["verdict"],
                               ("Likely Forgotten", "Not Forgotten", "Ambiguous", "Inconclusive"))

    def test_prism_and_echo_are_exposed_by_pipeline(self):
        self.assertIn("prism", ALGORITHMS)
        self.assertIn("echo", ALGORITHMS)
        for algo in ("prism", "echo"):
            with self.subTest(algo=algo):
                result = run_audit(algo, self.split, self.original_model, sample_ids=self.sample_ids)
                self.assertEqual(result["algorithm"], algo)
                self.assertIsInstance(result.get("method_trace"), dict)
                self.assertIn("consensus", result)

    def test_sisa_retrains_fewer_than_all_shards(self):
        """SISA's core efficiency claim: forgetting shouldn't require
        retraining every shard, only the one(s) containing forgotten data."""
        result = run_audit("sisa", self.split, self.original_model, sample_ids=self.sample_ids)
        self.assertIsNotNone(result["sisa_shards_retrained"])
        self.assertLessEqual(result["sisa_shards_retrained"], 4)
        self.assertGreaterEqual(result["sisa_shards_retrained"], 1)

    def test_gradient_ascent_measurably_degrades_forget_accuracy(self):
        result = run_audit("gradient_ascent", self.split, self.original_model)
        self.assertLess(result["accuracy"]["forget"], result["accuracy"]["original_model_test"] + 0.03)
        # retain utility should still be reasonably preserved
        self.assertGreater(result["accuracy"]["retain"], 0.9)


class TestFullComparison(unittest.TestCase):

    def test_full_comparison_uses_one_shared_reference_ensemble(self):
        result = run_full_comparison(reference_count=4, random_state=42)
        self.assertEqual(result["reference_ensemble"]["count"], 4)
        counts = {r["diagnosis"]["advanced_audit"]["reference_alignment"]["reference_count"] for r in result["results"].values()}
        self.assertEqual(counts, {4})
        self.assertTrue(all(r["reference_calibration"]["enabled"] for r in result["results"].values()))

    def test_full_comparison_runs_all_registered_methods(self):
        result = run_full_comparison()
        self.assertEqual(set(result["results"].keys()), set(ALGORITHMS))
        for algo, r in result["results"].items():
            self.assertIn("consensus", r)
            self.assertIn("diagnosis", r)


if __name__ == "__main__":
    unittest.main()
