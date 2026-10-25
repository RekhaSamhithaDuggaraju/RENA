import unittest
from core.llm.audit_controller import AuditPolicy, run_adaptive_audit


class AdaptiveAuditTests(unittest.TestCase):
    def test_confident_initial_evidence_skips_escalation(self):
        called = []
        decision = run_adaptive_audit(
            # Illustrative distinct forgetting-probe families for this generic
            # controller test; production independence must be justified separately.
            {"a": lambda: {"score": .92, "evidence_group": "forget_probe_family_a"},
             "b": lambda: {"score": .88, "evidence_group": "forget_probe_family_b"}},
            {"recovery": lambda: called.append(True) or {"score": .9}},
        )
        self.assertEqual(decision.status, "SUPPORTED")
        self.assertIsNone(decision.confidence)
        self.assertAlmostEqual(decision.support_score_mean, .9)
        self.assertFalse(decision.escalated)
        self.assertEqual(called, [])

    def test_uncertain_evidence_escalates(self):
        decision = run_adaptive_audit(
            {"a": lambda: {"score": .51, "evidence_group": "forget"}},
            {"recovery": lambda: {"score": .2, "evidence_group": "recovery"}},
        )
        self.assertTrue(decision.escalated)
        self.assertEqual(len(decision.evidence), 2)
        self.assertEqual(decision.evidence[-1]["stage"], "escalation")

    def test_disagreement_abstains(self):
        decision = run_adaptive_audit(
            {"a": lambda: {"score": .95, "evidence_group": "a"},
             "b": lambda: {"score": .1, "evidence_group": "b"}}, {},
        )
        self.assertEqual(decision.status, "ABSTAIN")

    def test_no_evidence_is_insufficient(self):
        self.assertEqual(run_adaptive_audit({}, {}).status, "INSUFFICIENT_EVIDENCE")

    def test_single_independent_group_is_insufficient(self):
        decision = run_adaptive_audit(
            {"a": lambda: {"score": .99, "evidence_group": "same"},
             "b": lambda: {"score": .98, "evidence_group": "same"}}, {},
        )
        self.assertEqual(decision.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(decision.independent_group_count, 1)
        self.assertIsNone(decision.confidence)
        self.assertAlmostEqual(decision.support_score_mean, .985)

    def test_weighted_group_aggregation(self):
        decision = run_adaptive_audit(
            {"a": lambda: {"score": .9, "reliability": 1, "evidence_group": "g1"},
             "b": lambda: {"score": .1, "reliability": 1, "evidence_group": "g1"},
             "c": lambda: {"score": .8, "reliability": .5, "evidence_group": "g2"}}, {},
        )
        # Same-group signals are averaged before groups are combined.
        # The resulting support score is not a calibrated confidence probability.
        self.assertAlmostEqual(decision.support_score_mean, .6)
        self.assertIsNone(decision.confidence)

    def test_forced_sentinel_escalation_runs_even_for_confident_initial(self):
        decision = run_adaptive_audit(
            {"a": lambda: {"score": .95, "evidence_group": "a"},
             "b": lambda: {"score": .92, "evidence_group": "b"}},
            {"recovery": lambda: {"score": .9, "evidence_group": "recovery"}},
            force_escalation=True,
        )
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.escalation_reason, "sentinel_escalation")

    def test_escalation_check_runs_once(self):
        calls = []
        def check():
            calls.append(1)
            return {"score": .8, "evidence_group": "extra"}
        run_adaptive_audit({"a": lambda: {"score": .5, "evidence_group": "a"}}, {"b": check})
        self.assertEqual(len(calls), 1)

    def test_invalid_score_and_reliability_rejected(self):
        for payload in ({"score": 1.5}, {"score": float("nan")}, {"score": .5, "reliability": -0.1}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                run_adaptive_audit({"bad": lambda p=payload: p}, {})

    def test_policy_threshold_validation(self):
        with self.assertRaises(ValueError):
            AuditPolicy(support_threshold=.2, reject_threshold=.4)


if __name__ == "__main__":
    unittest.main()
