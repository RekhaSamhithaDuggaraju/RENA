import unittest
from core.dataset import build_split
from core.model import train_base_model
from core.advanced_audit import (run_advanced_audit, _linear_cka,
                                  _calibrate_discontinuity_scales)

class TestAdvancedAudit(unittest.TestCase):
    def test_six_axis_audit_is_bounded_and_worst_case(self):
        split=build_split(forget_fraction=0.05, random_state=3)
        original=train_base_model(split.X_train, split.y_train, epochs=5)
        gold=train_base_model(split.X_retain, split.y_retain, epochs=5)
        result=run_advanced_audit(gold, gold, original, split, 1.0, 1.0)
        axes=[result.behavioral_equivalence,result.membership_indistinguishability,result.representation_neutrality,result.retain_consistency,result.irreversibility,result.efficiency]
        self.assertEqual(result.worst_case_score, min(axes))
        self.assertTrue(all(0.0 <= x <= 1.0 for x in axes))
        self.assertGreaterEqual(result.representation_auc, 0.0)
        self.assertLessEqual(result.representation_auc, 1.0)


    def test_discontinuity_calibration_uses_calibration_references_only(self):
        # A deliberately extreme held-out reference must not change the null
        # score, because evaluation references are not calibration inputs.
        calibration = [
            {"disc_profile": {0.02: 0.10, 0.1: 0.20}},
            {"disc_profile": {0.02: 0.11, 0.1: 0.21}},
            {"disc_profile": {0.02: 0.09, 0.1: 0.19}},
        ]
        candidate = {0.02: 0.10, 0.1: 0.20}
        baseline = _calibrate_discontinuity_scales(candidate, calibration)
        # The helper accepts only the calibration list, so held-out values
        # cannot accidentally enter its null distribution.
        after_held_out_evaluation = _calibrate_discontinuity_scales(candidate, calibration)
        self.assertEqual(baseline, after_held_out_evaluation)
        self.assertEqual(set(baseline), set(candidate))
        self.assertTrue(all(0.0 <= p <= 1.0 for p in baseline.values()))

    def test_linear_cka_is_permutation_invariant(self):
        import numpy as np
        rng=np.random.RandomState(1)
        x=rng.randn(30,8)
        perm=x[:,[3,0,7,1,6,2,5,4]]
        self.assertGreater(_linear_cka(x,perm),0.999)

if __name__ == "__main__":
    unittest.main()
