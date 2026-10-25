import unittest
import numpy as np

from core.advanced_audit import _calibrated_score, _two_sided_log_p
from core.dataset import build_split
from core.model import train_base_model
from core.probes import discontinuity_profile, discontinuity_score
from core.research_suite import _build_output_patch_controls
from core.unlearning import ground_truth_retrain


class TestCalibrationResolution(unittest.TestCase):
    def test_axis_can_reject_with_small_null(self):
        # 15 null pairs: the plain empirical p-value floor is 1/16 = 0.0625 (> 0.05),
        # so before V15 no axis could ever reject. Tail extrapolation fixes that.
        null = np.array([0.010, 0.012, 0.011, 0.013, 0.009, 0.012, 0.010, 0.011,
                         0.014, 0.012, 0.010, 0.013, 0.011, 0.012, 0.010])
        self.assertLess(_calibrated_score(0.5, null), 0.05)

    def test_in_range_values_keep_empirical_p(self):
        null = np.linspace(0.01, 0.02, 15)
        self.assertAlmostEqual(_calibrated_score(0.015, null), (np.sum(null >= 0.015) + 1) / 16)

    def test_two_sided_p_rejects_both_tails(self):
        null = [0.0075, 0.0080, 0.0078, 0.0082, 0.0076, 0.0079, 0.0081, 0.0077]
        self.assertLess(_two_sided_log_p(0.1, null), 0.01)      # too jumpy
        self.assertLess(_two_sided_log_p(0.0, null), 0.01)      # too flat
        self.assertGreater(_two_sided_log_p(0.0079, null), 0.5)  # typical


class TestDiscontinuityProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(0.10, 42)
        X = np.vstack([cls.split.X_retain, cls.split.X_forget])
        y = np.concatenate([cls.split.y_retain, cls.split.y_forget])
        cls.original = train_base_model(X, y)
        cls.refs = [ground_truth_retrain(cls.split.X_retain, cls.split.y_retain, random_state=900 + i)
                    for i in range(3)]

    def test_honest_models_are_smooth(self):
        prof = discontinuity_profile(self.refs[0], self.split.X_forget, self.split.y_forget)
        self.assertLess(prof[0.02], 0.02)

    def test_exact_input_patch_is_discontinuous(self):
        label, patched, *_ = _build_output_patch_controls(self.original, self.refs, self.split,
                                                          suppressions=(0.03,))[0]
        honest = discontinuity_score(discontinuity_profile(self.refs[0], self.split.X_forget, self.split.y_forget))
        attack = discontinuity_score(discontinuity_profile(patched, self.split.X_forget, self.split.y_forget))
        self.assertGreater(attack, 5 * honest)


if __name__ == "__main__":
    unittest.main()
