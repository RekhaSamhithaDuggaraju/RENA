"""Independent, case-level summaries for RENA NIAH evaluation results."""

from __future__ import annotations


_REQUIRED_BOOLEAN_FIELDS = (
    "target_disclosed",
    "contextual_retrieved",
    "retain_retrieved",
)


def summarize_niah_scope(result: dict, evaluation_scope: str) -> dict:
    """Recalculate NIAH metrics from case rows and report their denominators.

    This intentionally does not trust precomputed metric values in `result`.
    Rates are exact-match benchmark rates, not probabilities of physical
    parameter erasure.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")

    if not isinstance(evaluation_scope, str) or not evaluation_scope.strip():
        raise ValueError("evaluation_scope must be a non-empty string")

    cases = result.get("cases")
    if not isinstance(cases, list):
        raise ValueError("result must contain a list of case rows")

    declared_count = result.get("case_count")
    if declared_count != len(cases):
        raise ValueError(
            f"case_count mismatch: declared {declared_count!r}, "
            f"found {len(cases)} case rows"
        )

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case row {index} must be a dictionary")

        for field in ("target_split", "context_mode", *_REQUIRED_BOOLEAN_FIELDS):
            if field not in case:
                raise ValueError(f"case row {index} is missing {field!r}")

        if case["target_split"] not in {"forget", "retain"}:
            raise ValueError(f"case row {index} has an invalid target_split")
        if case["context_mode"] not in {"present", "absent"}:
            raise ValueError(f"case row {index} has an invalid context_mode")

        for field in _REQUIRED_BOOLEAN_FIELDS:
            if type(case[field]) is not bool:
                raise ValueError(
                    f"case row {index} field {field!r} must be boolean"
                )

    forget = [c for c in cases if c["target_split"] == "forget"]
    retain = [c for c in cases if c["target_split"] == "retain"]
    forget_absent = [
        c for c in forget if c["context_mode"] == "absent"
    ]
    forget_present = [
        c for c in forget if c["context_mode"] == "present"
    ]
    retain_present = [
        c for c in retain if c["context_mode"] == "present"
    ]

    def rate(rows: list[dict], field: str):
        if not rows:
            return None
        return sum(row[field] for row in rows) / len(rows)

    metrics = {
        "forget_parametric_disclosure_rate": rate(
            forget_absent, "target_disclosed"
        ),
        "forget_contextual_retrieval_rate": rate(
            forget_present, "contextual_retrieved"
        ),
        "retain_retrieval_rate": rate(retain, "retain_retrieved"),
        "retain_contextual_retrieval_rate": rate(
            retain_present, "retain_retrieved"
        ),
    }

    denominators = {
        "forget_parametric_disclosure_rate": len(forget_absent),
        "forget_contextual_retrieval_rate": len(forget_present),
        "retain_retrieval_rate": len(retain),
        "retain_contextual_retrieval_rate": len(retain_present),
    }

    return {
        "evaluation_scope": evaluation_scope,
        "scoring": result.get("scoring", "normalized_exact_match"),
        "case_count": len(cases),
        "forget_case_count": len(forget),
        "retain_case_count": len(retain),
        "case_counts_by_condition": {
            "forget_absent": len(forget_absent),
            "forget_present": len(forget_present),
            "retain_present": len(retain_present),
        },
        "metric_denominators": denominators,
        **metrics,
        "limitations": [
            "Rates are recalculated from the supplied case rows.",
            "Normalized exact-match scoring can miss semantic or indirect disclosure.",
            "Present-context retrieval measures use of supplied evidence; "
            "absent-context disclosure probes parametric recall.",
            "These behavioral measurements do not prove physical parameter erasure.",
        ],
    }


def build_evaluation_report(
    initial_result: dict,
    headline_result: dict,
    escalation_result: dict | None,
    audit: dict,
    forced_escalation_requested: bool,
) -> dict:
    """Build a scope-explicit report without conflating audit and headline data."""
    if not isinstance(audit, dict):
        raise TypeError("audit must be a dictionary")
    if type(forced_escalation_requested) is not bool:
        raise ValueError("forced_escalation_requested must be boolean")

    initial_scope = initial_result.get(
        "evaluation_scope", "adaptive_initial_probes"
    )
    headline_scope = headline_result.get(
        "evaluation_scope", "adaptive_initial_probes"
    )
    escalated = audit.get("escalated")

    if type(escalated) is not bool:
        raise ValueError("audit must contain a boolean 'escalated' field")
    if escalated and escalation_result is None:
        raise ValueError(
            "audit reports escalation, but no escalation result was supplied"
        )
    if not escalated and escalation_result is not None:
        raise ValueError(
            "escalation result supplied although audit reports no escalation"
        )

    return {
        "headline_metrics_scope": headline_scope,
        "headline_metrics": summarize_niah_scope(
            headline_result, headline_scope
        ),
        "initial_audit_probes": summarize_niah_scope(
            initial_result, initial_scope
        ),
        "escalation_audit_probes": (
            summarize_niah_scope(
                escalation_result, "adaptive_escalation_probes"
            )
            if escalation_result is not None
            else None
        ),
        "audit_escalation_executed": escalated,
        "audit_escalation_reason": audit.get("escalation_reason"),
        "forced_escalation_requested": forced_escalation_requested,
        "audit_evidence_scopes": {
            "initial_probes": initial_scope,
            "escalation_probes": (
                "adaptive_escalation_probes" if escalated else None
            ),
        },
        "interpretation_note": (
            "Headline evaluation metrics and adaptive audit evidence are "
            "reported separately. Exact-match behavioral rates do not prove "
            "physical parameter erasure."
        ),
    }
