import unittest

from core.llm.niah_reporting import summarize_niah_scope


def make_case(split, mode, disclosed=False, contextual=False, retained=False):
    return {
        "target_split": split,
        "context_mode": mode,
        "target_disclosed": disclosed,
        "contextual_retrieved": contextual,
        "retain_retrieved": retained,
    }


class NiahReportingTests(unittest.TestCase):
    def test_recalculates_metrics_from_case_rows_not_stored_rates(self):
        cases = [
            make_case("forget", "absent", disclosed=True),
            make_case("forget", "absent", disclosed=False),
            make_case("forget", "present", contextual=True),
            make_case("forget", "present", contextual=False),
            make_case("retain", "present", retained=True),
            make_case("retain", "present", retained=False),
            make_case("retain", "absent", retained=True),
            make_case("retain", "absent", retained=False),
        ]
        result = {
            "case_count": len(cases),
            "cases": cases,
            # Deliberately false values: the helper must not trust these.
            "forget_parametric_disclosure_rate": 0.99,
            "forget_contextual_retrieval_rate": 0.99,
            "retain_retrieval_rate": 0.99,
            "retain_contextual_retrieval_rate": 0.99,
        }

        report = summarize_niah_scope(result, "test_scope")

        self.assertEqual(report["case_count"], 8)
        self.assertEqual(report["forget_case_count"], 4)
        self.assertEqual(report["retain_case_count"], 4)
        self.assertEqual(report["forget_parametric_disclosure_rate"], 0.5)
        self.assertEqual(report["forget_contextual_retrieval_rate"], 0.5)
        self.assertEqual(report["retain_retrieval_rate"], 0.5)
        self.assertEqual(report["retain_contextual_retrieval_rate"], 0.5)
        self.assertEqual(
            report["metric_denominators"],
            {
                "forget_parametric_disclosure_rate": 2,
                "forget_contextual_retrieval_rate": 2,
                "retain_retrieval_rate": 4,
                "retain_contextual_retrieval_rate": 2,
            },
        )

    def test_empty_metric_subgroup_is_none_not_zero(self):
        cases = [
            make_case("forget", "absent", disclosed=False),
            make_case("retain", "absent", retained=True),
        ]
        report = summarize_niah_scope(
            {"case_count": 2, "cases": cases}, "small_test"
        )

        self.assertEqual(report["forget_parametric_disclosure_rate"], 0.0)
        self.assertIsNone(report["forget_contextual_retrieval_rate"])
        self.assertEqual(report["retain_retrieval_rate"], 1.0)
        self.assertIsNone(report["retain_contextual_retrieval_rate"])
        self.assertEqual(
            report["metric_denominators"]["retain_contextual_retrieval_rate"],
            0,
        )

    def test_detects_case_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "case_count mismatch"):
            summarize_niah_scope(
                {"case_count": 2, "cases": [make_case("forget", "absent")]},
                "test_scope",
            )

    def test_rejects_missing_metric_fields(self):
        with self.assertRaisesRegex(ValueError, "target_disclosed"):
            summarize_niah_scope(
                {
                    "case_count": 1,
                    "cases": [{
                        "target_split": "forget",
                        "context_mode": "absent",
                        "contextual_retrieved": False,
                        "retain_retrieved": False,
                    }],
                },
                "test_scope",
            )

    def test_rejects_non_boolean_metric_values(self):
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            summarize_niah_scope(
                {
                    "case_count": 1,
                    "cases": [{
                        "target_split": "forget",
                        "context_mode": "absent",
                        "target_disclosed": 1,
                        "contextual_retrieved": False,
                        "retain_retrieved": False,
                    }],
                },
                "test_scope",
            )

    def test_rejects_invalid_scope(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            summarize_niah_scope(
                {"case_count": 0, "cases": []}, " "
            )


if __name__ == "__main__":
    unittest.main()


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.initial = {
            "case_count": 2,
            "evaluation_scope": "adaptive_initial_probes",
            "cases": [
                make_case("forget", "absent", disclosed=False),
                make_case("retain", "present", retained=True),
            ],
        }
        self.full = {
            "case_count": 4,
            "evaluation_scope": "full_matched_grid",
            "cases": [
                make_case("forget", "absent", disclosed=False),
                make_case("forget", "present", contextual=True),
                make_case("retain", "present", retained=True),
                make_case("retain", "absent", retained=False),
            ],
        }
        self.escalation = {
            "case_count": 2,
            "cases": [
                make_case("forget", "absent", disclosed=True),
                make_case("retain", "present", retained=True),
            ],
        }

    def test_full_grid_headline_does_not_replace_initial_audit_scope(self):
        from core.llm.niah_reporting import build_evaluation_report

        report = build_evaluation_report(
            self.initial,
            self.full,
            None,
            {"escalated": False, "escalation_reason": None},
            False,
        )

        self.assertEqual(
            report["headline_metrics_scope"], "full_matched_grid"
        )
        self.assertEqual(report["headline_metrics"]["case_count"], 4)
        self.assertEqual(
            report["initial_audit_probes"]["evaluation_scope"],
            "adaptive_initial_probes",
        )
        self.assertEqual(report["initial_audit_probes"]["case_count"], 2)
        self.assertIsNone(report["escalation_audit_probes"])
        self.assertFalse(report["audit_escalation_executed"])
        self.assertIsNone(
            report["audit_evidence_scopes"]["escalation_probes"]
        )

    def test_escalation_metrics_and_execution_state_are_reported(self):
        from core.llm.niah_reporting import build_evaluation_report

        report = build_evaluation_report(
            self.initial,
            self.initial,
            self.escalation,
            {
                "escalated": True,
                "escalation_reason": "sentinel_escalation",
            },
            True,
        )

        self.assertTrue(report["audit_escalation_executed"])
        self.assertTrue(report["forced_escalation_requested"])
        self.assertEqual(
            report["audit_escalation_reason"], "sentinel_escalation"
        )
        self.assertEqual(
            report["escalation_audit_probes"]["case_count"], 2
        )
        self.assertEqual(
            report["escalation_audit_probes"][
                "forget_parametric_disclosure_rate"
            ],
            1.0,
        )
        self.assertEqual(
            report["audit_evidence_scopes"]["escalation_probes"],
            "adaptive_escalation_probes",
        )

    def test_rejects_escalated_audit_without_escalation_result(self):
        from core.llm.niah_reporting import build_evaluation_report

        with self.assertRaisesRegex(ValueError, "no escalation result"):
            build_evaluation_report(
                self.initial,
                self.initial,
                None,
                {"escalated": True},
                False,
            )

    def test_rejects_escalation_result_when_audit_did_not_escalate(self):
        from core.llm.niah_reporting import build_evaluation_report

        with self.assertRaisesRegex(ValueError, "although audit reports no"):
            build_evaluation_report(
                self.initial,
                self.initial,
                self.escalation,
                {"escalated": False},
                False,
            )
