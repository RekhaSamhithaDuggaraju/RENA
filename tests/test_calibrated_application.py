import unittest
import numpy as np
from core.dataset import build_split
from core.model import train_base_model
from core.pipeline import run_audit
from core.research_suite import adversarial_model_suite


class TestCalibratedApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split=build_split(.10,42)
        X=np.vstack([cls.split.X_retain,cls.split.X_forget])
        y=np.concatenate([cls.split.y_retain,cls.split.y_forget])
        cls.original=train_base_model(X,y)
        cls.ids=np.arange(len(X))

    def test_application_path_uses_reference_distribution_when_requested(self):
        result=run_audit("ground_truth_retraining",self.split,self.original,self.ids,
                         reference_count=4,reference_seed=65000,split_seed=42)
        audit=result["diagnosis"]["advanced_audit"]
        self.assertTrue(audit["reference_alignment"]["calibrated"])
        self.assertEqual(audit["reference_alignment"]["reference_count"],4)
        self.assertIn("certification_verdict", result["reference_calibration"]["calibration"])
        self.assertIn("provenance", result)
        self.assertEqual(result["provenance"]["split_seed"],42)

    def test_model_level_adversarial_controls_are_real_models(self):
        result=adversarial_model_suite(seed=42,forget_fraction=.10,reference_count=3)
        names={row["attack"] for row in result["controls"]}
        self.assertTrue({"partial_25","partial_50","partial_75","selective","over_unlearning","verifier_targeted"}.issubset(names))
        self.assertEqual(result["reference_count"],3)
        self.assertNotIn("score_corruption", names)


if __name__ == "__main__":
    unittest.main()
