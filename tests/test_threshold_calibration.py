import unittest
from core.threshold_calibration import select_development_thresholds


class TestThresholdCalibration(unittest.TestCase):
    def test_selects_thresholds_without_false_passes(self):
        rows = [
            {"label":"good", "correctness_min":.8, "forget_progress":.7, "retain_utility_ratio":.98},
            {"label":"good", "correctness_min":.7, "forget_progress":.6, "retain_utility_ratio":.96},
            {"label":"bad", "correctness_min":.8, "forget_progress":.05, "retain_utility_ratio":1.0},
            {"label":"bad", "correctness_min":.8, "forget_progress":.8, "retain_utility_ratio":.5},
        ]
        result = select_development_thresholds(rows, target_false_pass_rate=0.05)
        self.assertEqual(result["development_false_pass_rate"], 0.0)
        self.assertTrue(result["frozen_before_held_out_evaluation"])
        self.assertIn("forget_pass", result["thresholds"])


if __name__ == "__main__":
    unittest.main()
