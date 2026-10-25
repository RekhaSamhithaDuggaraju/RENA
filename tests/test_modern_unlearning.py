import unittest
import numpy as np
from core.dataset import build_split
from core.model import train_base_model
from core.modern_unlearning import negative_gradient_unlearn, salun_unlearn, unsir_style_unlearn, scrub_style_unlearn, ModernConfig


class TestModernUnlearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(forget_fraction=0.05, random_state=11)
        cls.model = train_base_model(cls.split.X_train, cls.split.y_train, epochs=10)
        cls.cfg = ModernConfig(forget_epochs=1, recovery_epochs=1)

    def test_modern_methods_change_parameters(self):
        s=self.split
        for fn in (negative_gradient_unlearn, salun_unlearn, unsir_style_unlearn, scrub_style_unlearn):
            with self.subTest(method=fn.__name__):
                m, trace = fn(self.model, s.X_forget, s.y_forget, s.X_retain, s.y_retain, self.cfg)
                self.assertFalse(np.allclose(m.parameter_vector(), self.model.parameter_vector()))
                self.assertIn("family", trace)
                self.assertTrue(np.all(np.isfinite(m.parameter_vector())))


if __name__ == "__main__":
    unittest.main()
