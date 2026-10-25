import unittest
from core.research_suite import threshold_sensitivity, confidence_verdict, verifier_ground_truth_benchmark, cost_quality_profile

class TestResearchExtensions(unittest.TestCase):
    def setUp(self):
        self.result={
            "module_results":[
                {"module_name":"A","forget_score":.9,"reliability":1.0,"p_value":.01,"reliability_reason":"ok","detail":""},
                {"module_name":"B","forget_score":.88,"reliability":1.0,"p_value":.02,"reliability_reason":"ok","detail":""},
                {"module_name":"C","forget_score":.91,"reliability":1.0,"p_value":.03,"reliability_reason":"ok","detail":""},
            ],
            "consensus":{"agreement":.95},
            "diagnosis":{"advanced_audit":{"worst_case_score":.82}},
        }
    def test_threshold_sensitivity(self):
        rows=threshold_sensitivity(self.result)
        self.assertEqual(len(rows),5)
        self.assertEqual(rows[0]["verdict"],"Likely Forgotten")
    def test_confidence(self):
        out=confidence_verdict(self.result)
        self.assertGreater(out["confidence"],.8)
        self.assertEqual(out["abstained"],False)
    def test_ground_truth_benchmark(self):
        out=verifier_ground_truth_benchmark()
        self.assertEqual(out["accuracy"],1.0)
    def test_cost_quality(self):
        out=cost_quality_profile([{"algorithm":"a","rena_x_worst_case":.8,"runtime_seconds":2},{"algorithm":"b","rena_x_worst_case":.7,"runtime_seconds":1}])
        self.assertGreater(out[0]["quality_per_second"],0)

if __name__=='__main__': unittest.main()
