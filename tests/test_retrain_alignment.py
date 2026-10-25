import tempfile
import unittest
from pathlib import Path
import numpy as np

from core.checkpoints import load_model_state, save_model_state
from core.consensus import consensus_baselines
from core.nn import SimpleMLP
from core.retrain_alignment import js_divergence, relative_retrain_alignment, stable_probabilities
from core.verification import ModuleResult


class RetrainAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., 1.], [1., 0.], [2., 1.]])
        self.original = SimpleMLP(2, 2, hidden_sizes=(3,), random_state=1)
        self.retrained = SimpleMLP(2, 2, hidden_sizes=(3,), random_state=2)

    def test_identical_audited_retrained_has_high_rra(self):
        self.assertGreater(relative_retrain_alignment(self.original, self.retrained, self.retrained, self.x)["rra"], .999)

    def test_identical_audited_original_has_low_rra(self):
        self.assertLess(relative_retrain_alignment(self.original, self.original, self.retrained, self.x)["rra"], .001)

    def test_stability_and_denominator_edge_case(self):
        value = relative_retrain_alignment(self.original, self.original, self.original, self.x)
        self.assertTrue(np.isfinite(value["rra"])); self.assertEqual(value["rra"], 0.0)
        self.assertTrue(np.allclose(stable_probabilities(np.array([[0., 0.], [1., 0.]] )).sum(axis=1), 1))
        self.assertTrue(np.all(np.isfinite(js_divergence(np.array([[0., 1.]]), np.array([[1., 0.]])))) )

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "model"; save_model_state(self.original, p, {"method": "test"}); restored = load_model_state(p)
            self.assertTrue(np.allclose(self.original.predict_proba(self.x), restored.predict_proba(self.x)))

    def test_baseline_modes_preserve_override_and_weighting(self):
        results = [ModuleResult("a", 0, None, .9, 1, "test", {}), ModuleResult("b", 0, None, .1, 1, "test", {})]
        modes = consensus_baselines(results)
        self.assertEqual(modes["RENA_FULL"]["verdict"], "Inconclusive")
        self.assertNotEqual(modes["WEIGHTED_NO_OVERRIDE"]["verdict"], "Inconclusive")


if __name__ == "__main__": unittest.main()
