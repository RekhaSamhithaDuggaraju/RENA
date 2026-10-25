import unittest
from core.llm.audit_benchmark import benchmark_audit_policies, generate_scenarios

class AuditBenchmarkTests(unittest.TestCase):
    def test_scenario_generation_is_reproducible_and_balanced(self):
        a = generate_scenarios(count=40, seed=9)
        self.assertEqual(a, generate_scenarios(count=40, seed=9))
        self.assertEqual(sum(x["truth_supported"] for x in a), 20)

    def test_benchmark_compares_three_policies(self):
        result = benchmark_audit_policies(count=80, seed=3)
        self.assertEqual(set(result["policies"]), {"adaptive", "adaptive_sentinel", "cheap_only", "fixed_all"})
        for row in result["policies"].values():
            self.assertEqual(row["metrics"]["n"], 80)
            self.assertGreaterEqual(row["metrics"]["coverage"], 0)
            self.assertLessEqual(row["metrics"]["coverage"], 1)

    def test_fixed_all_has_expected_assumed_cost(self):
        result = benchmark_audit_policies(count=40, seed=1)
        self.assertEqual(result["policies"]["fixed_all"]["metrics"]["mean_cost_units"], 10.0)
        self.assertEqual(result["policies"]["cheap_only"]["metrics"]["mean_cost_units"], 2.0)

    def test_sentinel_policy_is_reproducible(self):
        a = benchmark_audit_policies(count=80, seed=3)
        b = benchmark_audit_policies(count=80, seed=3)
        self.assertEqual(a["policies"]["adaptive_sentinel"], b["policies"]["adaptive_sentinel"])
        self.assertGreater(a["policies"]["adaptive_sentinel"]["metrics"]["escalation_rate"], 0)

    def test_rejects_too_few_cases(self):
        with self.assertRaises(ValueError):
            benchmark_audit_policies(count=5)

if __name__ == "__main__": unittest.main()
