import unittest
from core.benchmark_compare import ultra_style_baseline
from core.research_suite import partial_forgetting_curve, verifier_benchmark_matrix

class BenchmarkMilestoneTests(unittest.TestCase):
    def test_ultra_style_baseline_is_explicit(self):
        r=ultra_style_baseline({"behavioral_equivalence":.9,"membership_indistinguishability":.8,"representation_neutrality":.7,"retain_consistency":.95,"irreversibility":.85,"efficiency":.5})
        self.assertEqual(r["score"],.5)
        self.assertTrue("not a claim of exact ULTRA reproduction" in r["reproduction_status"])

    def test_held_out_calibration_metadata(self):
        r=verifier_benchmark_matrix(seed=42,forget_fraction=.10,reference_count=6)
        c=r["calibration"]
        self.assertEqual(c["calibration_reference_count"],4)
        self.assertEqual(c["evaluation_reference_count"],2)
        self.assertEqual(c["mode"],"held_out_empirical_retrain_null")

    def test_partial_curve_has_five_conditions(self):
        r=partial_forgetting_curve(seed=42,forget_fraction=.10,reference_count=6)
        self.assertEqual([x["forgetting_fraction"] for x in r["rows"]],[0.0,.25,.5,.75,1.0])

if __name__=='__main__': unittest.main()
