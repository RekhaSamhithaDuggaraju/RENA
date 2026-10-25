"""
Tests for the Consensus Engine — this is the module doing the intellectual
work, so it gets the most scrutiny. The adversarial disagreement test is
the one that proves the engine resists being fooled by a high average score
when the underlying modules don't actually agree.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from core.verification import ModuleResult
from core.consensus import compute_consensus, resolve_verdict


def make_result(name, forget_score, reliability=1.0, reason="test"):
    return ModuleResult(
        module_name=name, raw_statistic=0.0, p_value=None,
        forget_score=forget_score, reliability=reliability,
        reliability_reason=reason, detail={},
    )


class TestConsensusEngine(unittest.TestCase):

    def test_all_modules_agree_high_score_gives_confident_verdict(self):
        results = [make_result(f"m{i}", 0.9) for i in range(4)]
        c = compute_consensus(results)
        self.assertGreater(c.agreement, 0.7)
        self.assertEqual(c.verdict, "Likely Forgotten")

    def test_all_modules_agree_low_score_gives_not_forgotten(self):
        results = [make_result(f"m{i}", 0.1) for i in range(4)]
        c = compute_consensus(results)
        self.assertGreater(c.agreement, 0.7)
        self.assertEqual(c.verdict, "Not Forgotten")

    def test_adversarial_disagreement_does_not_produce_false_confidence(self):
        """
        THE key test: 3 modules say "forgotten" (~0.9), 1 says "not forgotten"
        (~0.1), all equal reliability. Naive averaging would give ~0.7 and
        could be read as a confident-ish "forgotten" result. The consensus
        engine MUST NOT return "Likely Forgotten" here -- disagreement this
        large must force Inconclusive or Ambiguous, never false confidence.
        """
        results = [
            make_result("mia", 0.92),
            make_result("loss_ks", 0.88),
            make_result("bootstrap", 0.85),
            make_result("confidence", 0.10),
        ]
        c = compute_consensus(results)
        self.assertNotEqual(c.verdict, "Likely Forgotten",
                             f"Engine gave false confidence despite disagreement: {c}")
        self.assertIn(c.verdict, ("Inconclusive", "Ambiguous"))

    def test_low_reliability_module_has_less_influence(self):
        """A module with low reliability should pull the consensus score less
        than a module with high reliability, holding forget_score equal."""
        high_rel = [make_result("a", 0.9, reliability=1.0), make_result("b", 0.1, reliability=1.0)]
        low_rel_outlier = [make_result("a", 0.9, reliability=1.0), make_result("b", 0.1, reliability=0.2)]
        c1 = compute_consensus(high_rel)
        c2 = compute_consensus(low_rel_outlier)
        # when b's reliability is discounted, consensus_score should move
        # closer to a's score (0.9) than in the equal-reliability case
        self.assertGreater(c2.consensus_score, c1.consensus_score)

    def test_small_forget_set_reduces_reliability(self):
        from core.verification import _reliability_from_sample_size
        rel_small, reason_small = _reliability_from_sample_size(5)
        rel_large, reason_large = _reliability_from_sample_size(500)
        self.assertLess(rel_small, rel_large)
        self.assertIn("below", reason_small)

    def test_resolve_verdict_boundary_conditions(self):
        self.assertEqual(resolve_verdict(0.9, 0.4), "Inconclusive")   # agreement too low
        self.assertEqual(resolve_verdict(0.8, 0.8), "Likely Forgotten")
        self.assertEqual(resolve_verdict(0.2, 0.8), "Not Forgotten")
        self.assertEqual(resolve_verdict(0.5, 0.8), "Ambiguous")      # score in dead zone

    def test_discounted_modules_are_surfaced_in_explanation(self):
        results = [make_result("a", 0.9, reliability=0.5, reason="small forget set")]
        c = compute_consensus(results)
        self.assertEqual(len(c.discounted_modules), 1)
        self.assertEqual(c.discounted_modules[0]["module"], "a")


if __name__ == "__main__":
    unittest.main()
