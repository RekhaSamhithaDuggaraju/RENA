import unittest
import numpy as np
from unittest.mock import patch
from urllib import error

from core.dataset import build_split
from core.model import train_base_model
from core.evidence import EvidenceItem, detect_contradictions, evidence_from_audit
from core.recovery_stress_test import CONDITIONS, run_recovery_stress_test, transform_images
from core.conformal_diagnostic import run_conformal_diagnostic
from core.llm_evidence_analyzer import LLMAPIError, PROVIDER, _post_gemini_api, analyze_evidence, fallback_analysis, validate_analysis


class TestEvidence(unittest.TestCase):
    def test_schema_serialization(self):
        item = EvidenceItem("E001", "test", "score", 0.9, 0.9, 1.0, True, [], [], {})
        self.assertEqual(item.to_dict()["evidence_id"], "E001")

    def test_contradiction_detection(self):
        items = [EvidenceItem("E001", "a", "s", 0.9, 0.9, 1, True, [], [], {}), EvidenceItem("E002", "b", "s", 0.1, 0.1, 1, False, [], [], {})]
        detect_contradictions(items)
        self.assertEqual(items[0].contradictions, ["E002"])
        self.assertEqual(items[1].contradictions, ["E001"])


class TestAnalyzer(unittest.TestCase):
    def test_fallback_is_explicit(self):
        result = fallback_analysis([{"evidence_id": "E001", "supports_forgetting": True}])
        self.assertEqual(result["analysis_mode"], "FALLBACK")
        self.assertIsNone(result["model"])
        self.assertEqual(validate_analysis(result)["overall_status"], "SUPPORTED")

    def test_invalid_llm_json_is_rejected(self):
        with self.assertRaises(ValueError): validate_analysis({})

    def test_api_failure_is_not_converted_to_fallback(self):
        http_error = error.HTTPError("https://generativelanguage.googleapis.com/v1beta/interactions", 401, "Unauthorized", {}, None)
        with patch("core.llm_evidence_analyzer.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(LLMAPIError, "HTTP 401"):
                _post_gemini_api("not-a-real-key", b"{}", attempts=1)

    def test_key_configured_propagates_api_failure(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-a-real-key"}, clear=True):
            with patch("core.llm_evidence_analyzer._post_gemini_api", side_effect=LLMAPIError("HTTP 429")):
                with self.assertRaisesRegex(LLMAPIError, "HTTP 429"):
                    analyze_evidence([])

    def test_valid_api_response_is_marked_real_api(self):
        payload = {"overall_status": "SUPPORTED", "supporting_evidence_ids": ["E001"], "contradictory_evidence_ids": [],
                   "missing_evidence": [], "limitations": [], "claim_strength": "MODERATE", "claim": "Test claim", "reasoning_summary": "Test summary"}
        response = {"steps": [{"type": "model_output", "content": [{"type": "text", "text": __import__("json").dumps(payload)}]}]}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-a-real-key"}, clear=True):
            with patch("core.llm_evidence_analyzer._post_gemini_api", return_value=response):
                result = analyze_evidence([])
        self.assertEqual(result["analysis_mode"], "REAL_API")
        self.assertEqual(result["provider"], PROVIDER)

    def test_malformed_gemini_response_is_validation_failure(self):
        response = {"output_text": '{"overall_status":"SUPPORTED"}'}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-a-real-key"}, clear=True):
            with patch("core.llm_evidence_analyzer._post_gemini_api", return_value=response):
                with self.assertRaisesRegex(LLMAPIError, "invalid structured output"):
                    analyze_evidence([])

    def test_gemini_cannot_introduce_unseen_metric_numbers(self):
        payload = {"overall_status": "SUPPORTED", "supporting_evidence_ids": [], "contradictory_evidence_ids": [],
                   "missing_evidence": [], "limitations": [], "claim_strength": "WEAK", "claim": "Accuracy was 99.9", "reasoning_summary": "Unsupported number."}
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-a-real-key"}, clear=True):
            with patch("core.llm_evidence_analyzer._post_gemini_api", return_value={"output_text": __import__("json").dumps(payload)}):
                with self.assertRaisesRegex(LLMAPIError, "invalid structured output"):
                    analyze_evidence([])


class TestDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = build_split(); cls.model = train_base_model(cls.split.X_train, cls.split.y_train, epochs=5)

    def test_recovery_stress_all_conditions_and_transform(self):
        records = run_recovery_stress_test(self.model, self.split)
        self.assertEqual({r["condition"] for r in records}, set(CONDITIONS))
        self.assertEqual(len(records), 10)
        self.assertTrue(all(0 <= r["normalized_recoverability"] <= 1 for r in records))
        self.assertEqual(transform_images(self.split.X_forget, "spatial_shift").shape, self.split.X_forget.shape)

    def test_conformal_diagnostic(self):
        result = run_conformal_diagnostic(self.model, self.split)
        self.assertIn("exclusion_gap", result)
        self.assertTrue(-1 <= result["exclusion_gap"] <= 1)


if __name__ == "__main__": unittest.main()
