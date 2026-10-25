import unittest
from pydantic import ValidationError
from backend.app.routers.llm_jobs import JobCreate

class LLMJobSchemaTests(unittest.TestCase):
    def test_accepts_valid_local_experiment_config(self):
        req = JobCreate(model="/models/small-lm", methods=["gradient_ascent"], seed=7, device="mps")
        self.assertEqual(req.model, "/models/small-lm")
        self.assertEqual(req.seed, 7)
        self.assertEqual(req.device, "mps")

    def test_rejects_invalid_numeric_bounds(self):
        with self.assertRaises(ValidationError):
            JobCreate(model="local", methods=["gradient_ascent"], epochs=0)
        with self.assertRaises(ValidationError):
            JobCreate(model="local", methods=["gradient_ascent"], max_new_tokens=999)

    def test_rejects_empty_methods(self):
        with self.assertRaises(ValidationError):
            JobCreate(model="local", methods=[])

if __name__ == "__main__":
    unittest.main()
