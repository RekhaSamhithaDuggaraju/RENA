import unittest
from core.llm.niah_audit_adapter import paired_niah_audit_decision


def aggregate(leak=.1, retain=.9, gap=.05, forget_present_gap=.0):
    return {
        "candidate": "GA",
        "reference": "retain_only_retraining",
        "seeds": [1, 2],
        "seed_count": 2,
        "metrics": {
            "forget_absent_disclosure": {
                "mean_candidate_rate": leak,
                "mean": gap,
                "case_counts": [10, 10],
            },
            "forget_present_retrieval": {
                "mean_candidate_rate": .9,
                "mean": forget_present_gap,
                "case_counts": [10, 10],
            },
            "retain_present_retrieval": {
                "mean_candidate_rate": retain,
                "mean": -gap,
                "case_counts": [10, 10],
            },
        },
    }


class NiahAuditAdapterTests(unittest.TestCase):
    def test_conservative_supported_case(self):
        out = paired_niah_audit_decision(aggregate())
        self.assertEqual(out["status"], "SUPPORTED")
        self.assertEqual(out["independent_group_count"], 1)
        self.assertEqual(out["benchmark_context"]["seed_count"], 2)

    def test_leakage_limits_composite(self):
        out = paired_niah_audit_decision(aggregate(leak=.8))
        self.assertEqual(out["status"], "NOT_SUPPORTED")
        self.assertIsNone(out["confidence"])
        self.assertAlmostEqual(out["support_score_mean"], .2)

    def test_requires_valid_rates(self):
        with self.assertRaises(ValueError):
            paired_niah_audit_decision(aggregate(leak=1.5))

    def test_refuses_missing_metrics(self):
        with self.assertRaises(ValueError):
            paired_niah_audit_decision({"metrics": {}})

    def test_forget_present_gap_affects_reference_alignment(self):
        baseline = paired_niah_audit_decision(aggregate())
        changed = paired_niah_audit_decision(
            aggregate(forget_present_gap=-.2)
        )

        baseline_details = baseline["evidence"][0]["details"]
        changed_details = changed["evidence"][0]["details"]

        self.assertAlmostEqual(
            baseline_details["mean_absolute_reference_gap"], .0333333333
        )
        self.assertAlmostEqual(
            changed_details["mean_absolute_reference_gap"], .1
        )
        self.assertLess(
            changed_details["component_support"]["reference_alignment"],
            baseline_details["component_support"]["reference_alignment"],
        )


if __name__ == "__main__":
    unittest.main()
