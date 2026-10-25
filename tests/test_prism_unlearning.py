import unittest
import numpy as np

from core.dataset import build_split
from core.model import train_base_model
from core.prism_unlearning import PrismConfig, _project_forget_gradient, prism_unlearn


class PrismUnlearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(random_state=42)
        cls.original = train_base_model(np.vstack([cls.split.X_retain, cls.split.X_forget]), np.r_[cls.split.y_retain, cls.split.y_forget])

    def test_projection_has_non_positive_retain_inner_product(self):
        _, _, details = _project_forget_gradient(self.original, self.split.X_forget, self.split.y_forget,
                                                  self.split.X_retain, self.split.y_retain, 1e-12)
        self.assertLessEqual(details["projected_retain_dot"], 1e-8)

    def test_prism_returns_finite_trace(self):
        model, trace = prism_unlearn(self.original, self.split, PrismConfig(iterations=2))
        self.assertEqual(trace["iterations"], 2)
        self.assertEqual(len(trace["trajectory"]), 2)
        self.assertTrue(all(np.all(np.isfinite(w)) for w in model.W))

