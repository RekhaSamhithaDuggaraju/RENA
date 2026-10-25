import unittest
import numpy as np

from core.dataset import build_split
from core.model import train_base_model
from core.unlearning import ground_truth_retrain
from core.advanced_audit import run_advanced_audit, _model_metrics
from core.adversarial_lab import verifier_targeted_deception, over_unlearning


class TestCertificationGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(forget_fraction=0.1, random_state=42)
        X = np.vstack([cls.split.X_retain, cls.split.X_forget])
        y = np.concatenate([cls.split.y_retain, cls.split.y_forget])
        cls.original = train_base_model(X, y)
        cls.refs = [ground_truth_retrain(cls.split.X_retain, cls.split.y_retain, random_state=50000+i) for i in range(10)]
        cls.ref_metrics = [_model_metrics(m, cls.split) for m in cls.refs]

    def audit(self, candidate):
        return run_advanced_audit(
            candidate, self.refs[0], self.original, self.split, 1.0, 1.0,
            reference_models=self.refs, reference_metrics=self.ref_metrics,
        )

    def test_exact_retraining_can_pass(self):
        audit = self.audit(self.refs[0])
        self.assertEqual(audit.calibration["certification_verdict"], "PASS")
        self.assertGreaterEqual(audit.forget_progress, 0.40)

    def test_no_op_cannot_pass_without_forget_progress(self):
        audit = self.audit(self.original)
        self.assertEqual(audit.calibration["certification_verdict"], "FAIL")
        self.assertLess(audit.forget_progress, 0.20)

    def test_verifier_targeted_surrogate_fails_forget_gate(self):
        candidate = verifier_targeted_deception(self.original, self.refs[0], self.split)
        audit = self.audit(candidate)
        self.assertEqual(audit.calibration["certification_verdict"], "FAIL")
        self.assertLess(audit.forget_progress, 0.20)

    def test_over_unlearning_fails_retain_utility_gate(self):
        candidate = over_unlearning(self.original, self.split)
        audit = self.audit(candidate)
        self.assertEqual(audit.calibration["certification_verdict"], "FAIL")
        self.assertLess(audit.retain_utility_ratio, 0.80)


if __name__ == "__main__":
    unittest.main()
