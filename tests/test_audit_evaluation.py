import unittest
from core.llm.audit_evaluation import evaluate_audit_decisions

class AuditEvaluationTests(unittest.TestCase):
    def test_perfect_decisions_and_coverage(self):
        rows = [
            {"case_id":"a", "status":"SUPPORTED", "truth_supported":True, "cost_units":2, "scenario_type":"clear"},
            {"case_id":"b", "status":"NOT_SUPPORTED", "truth_supported":False, "cost_units":3, "scenario_type":"clear"},
            {"case_id":"c", "status":"ABSTAIN", "truth_supported":True, "cost_units":1, "scenario_type":"hard"},
        ]
        r = evaluate_audit_decisions(rows, bootstrap_iterations=100)
        self.assertAlmostEqual(r["metrics"]["coverage"], 2/3)
        self.assertEqual(r["metrics"]["selective_risk"], 0)
        self.assertEqual(r["metrics"]["confusion"], {"tp":1,"fp":0,"tn":1,"fn":0})
        self.assertIn("hard", r["by_scenario"])

    def test_bad_labels_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_audit_decisions([{"case_id":"a", "status":"SUPPORTED", "truth_supported":1}])
        row = {"case_id":"a", "status":"SUPPORTED", "truth_supported":True}
        with self.assertRaises(ValueError):
            evaluate_audit_decisions([row, row])

    def test_bootstrap_is_reproducible(self):
        rows = [{"case_id":str(i), "status":"SUPPORTED" if i%2==0 else "NOT_SUPPORTED",
                 "truth_supported":i%2==0} for i in range(20)]
        a = evaluate_audit_decisions(rows, seed=7, bootstrap_iterations=100)
        b = evaluate_audit_decisions(rows, seed=7, bootstrap_iterations=100)
        self.assertEqual(a["bootstrap"], b["bootstrap"])

if __name__ == "__main__": unittest.main()
