"""Contract tests for the interactive classical audit request/response schema."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.schemas.audit import AuditCreateRequest, Algorithm
from pydantic import ValidationError


class AuditRequestSchemaTests(unittest.TestCase):
    def test_all_registered_methods_are_accepted(self):
        methods = ["fine_tuning", "gradient_ascent", "ground_truth_retraining", "sisa", "prism", "echo", "neggrad", "salun", "unsir", "scrub"]
        for method in methods:
            with self.subTest(method=method):
                self.assertEqual(AuditCreateRequest(algorithm=method).algorithm, method)

    def test_default_split_seed_and_fraction(self):
        request = AuditCreateRequest(algorithm="echo")
        self.assertEqual(request.split_seed, 42)
        self.assertEqual(request.forget_fraction, 0.1)

    def test_custom_split_seed(self):
        request = AuditCreateRequest(algorithm="prism", forget_fraction=0.2, split_seed=2026)
        self.assertEqual(request.split_seed, 2026)
        self.assertEqual(request.forget_fraction, 0.2)

    def test_invalid_seed_rejected(self):
        for seed in (-1, 2_147_483_648):
            with self.subTest(seed=seed), self.assertRaises(ValidationError):
                AuditCreateRequest(algorithm="sisa", split_seed=seed)

    def test_invalid_fraction_rejected(self):
        for fraction in (0, 0.6):
            with self.subTest(fraction=fraction), self.assertRaises(ValidationError):
                AuditCreateRequest(algorithm="fine_tuning", forget_fraction=fraction)


if __name__ == "__main__":
    unittest.main()
