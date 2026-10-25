import unittest
import numpy as np

from core.dataset import build_split
from core.model import train_base_model
from core.unlearning import ground_truth_retrain
from core.verifier_attack import rena_x_targeted_deception
from core.verifier_matrix import evaluate_candidate


class TestVerifierAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(forget_fraction=0.1, random_state=42)
        X = np.vstack([cls.split.X_retain, cls.split.X_forget])
        y = np.concatenate([cls.split.y_retain, cls.split.y_forget])
        cls.original = train_base_model(X, y)
        cls.refs = [ground_truth_retrain(cls.split.X_retain, cls.split.y_retain, random_state=61000+i)
                    for i in range(4)]

    def test_attack_queries_complete_verifier(self):
        result = rena_x_targeted_deception(self.original, self.refs, self.split, budget=2)
        self.assertEqual(result["threat_model"], "black_box_adaptive_query_attack")
        self.assertEqual(result["query_budget"], 2)
        self.assertIn(result["selected"]["verdict"], {"PASS", "REVIEW", "FAIL"})
        self.assertEqual(len(result["trials"]), 2)
        self.assertIsNotNone(result["candidate_model"])

    def test_matrix_has_all_comparators(self):
        result = rena_x_targeted_deception(self.original, self.refs, self.split, budget=1)
        row = evaluate_candidate("adaptive", result["candidate_model"], self.original,
                                 self.split, self.refs)
        for method in ("rena_x", "mia", "ks", "simple_average", "ultra_style"):
            self.assertIn("verdict", row[method])


if __name__ == "__main__":
    unittest.main()

class TestPipelineDefaults(unittest.TestCase):
    def test_run_audit_default_is_calibrated_reference_mode(self):
        import inspect
        from core.pipeline import run_audit
        default = inspect.signature(run_audit).parameters["reference_count"].default
        self.assertEqual(default, 10)
