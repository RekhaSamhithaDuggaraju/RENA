import unittest
from core.llm.statistics import aggregate_suite_summaries, summarize

class TestLLMStatistics(unittest.TestCase):
    def test_empty_metric(self):
        self.assertIsNone(summarize([None])["mean"])

    def test_summary_mean_and_seed_count(self):
        s = summarize([0.2, 0.4, 0.6], bootstrap_samples=200, seed=7)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 0.4)
        self.assertLessEqual(s["ci95"][0], s["mean"])
        self.assertGreaterEqual(s["ci95"][1], s["mean"])

    def test_paired_contrast_by_seed(self):
        def row(seed, ga, ref):
            return {"seed": seed, "base_model": "model-x", "protocol": "matched-v1", "methods": {
                "gradient_ascent": {"forget_parametric_disclosure_rate": ga},
                "retain_only_retraining": {"forget_parametric_disclosure_rate": ref}}}
        out = aggregate_suite_summaries([row(1, .2, .5), row(2, .3, .5)], bootstrap_samples=200)
        delta = out["paired_differences_vs_retain_only"]["gradient_ascent"]["forget_parametric_disclosure_rate"]
        self.assertEqual(delta["n"], 2)
        self.assertAlmostEqual(delta["mean"], -.25)

    def test_rejects_duplicate_seed(self):
        row = {"seed": 1, "base_model": "m", "protocol": "p", "methods": {}}
        with self.assertRaises(ValueError):
            aggregate_suite_summaries([row, row], bootstrap_samples=200)

if __name__ == "__main__": unittest.main()
