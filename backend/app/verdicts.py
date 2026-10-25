from __future__ import annotations


def _normalize(v: str | None) -> str:
    value=(v or 'REVIEW').strip().upper()
    if value in {'REVIEW','AMBIGUOUS','UNCERTAIN'}: return 'INCONCLUSIVE'
    if value in {'LIKELY FORGOTTEN'}: return 'PASS'
    if value in {'NOT FORGOTTEN'}: return 'FAIL'
    return value if value in {'PASS','FAIL','INCONCLUSIVE'} else 'INCONCLUSIVE'


def authoritative_verdict(audit):
    """Single application decision source: RENA-X certification verdict, normalized to PASS/FAIL/INCONCLUSIVE."""
    raw=(getattr(audit, 'authoritative_verdict', None)
         or getattr(audit, 'rena_x_verdict', None)
         or ((getattr(audit, 'diagnosis', None) or {}).get('advanced_audit') or {}).get('verdict')
         or getattr(audit, 'consensus_verdict', None))
    return _normalize(raw)


def authoritative_score(audit):
    v=getattr(audit, 'rena_x_score', None)
    if v is not None: return float(v)
    ax=((getattr(audit, 'diagnosis', None) or {}).get('advanced_audit') or {})
    v=ax.get('certification_score', ax.get('worst_case_score'))
    return float(v) if v is not None else None


def confidence_and_quality(audit):
    """Summarize evidence strength independently of the RENA-X score.

    This is a bounded evidence-strength indicator, not a probability that
    the unlearning verdict is correct.
    """
    ax = ((getattr(audit, "diagnosis", None) or {}).get("advanced_audit") or {})
    axes = ax.get("axes") or {}
    pvals = [
        float(item["p_value"])
        for item in axes.values()
        if isinstance(item, dict) and item.get("p_value") is not None
    ]

    summary = getattr(audit, "dataset_summary", None) or {}
    ref_count = int(summary.get("reference_retrains") or 0)
    seeds = summary.get("seeds") or []
    seed_count = len(seeds) or (
        1 if summary.get("split_seed") is not None else 0
    )

    authoritative = authoritative_verdict(audit)
    consensus_raw = str(getattr(audit, "consensus_verdict", "") or "").strip().upper()
    consensus_map = {
        "PASS": "PASS",
        "LIKELY FORGOTTEN": "PASS",
        "FAIL": "FAIL",
        "NOT FORGOTTEN": "FAIL",
        "INCONCLUSIVE": "INCONCLUSIVE",
        "AMBIGUOUS": "INCONCLUSIVE",
        "REVIEW": "INCONCLUSIVE",
        "UNCERTAIN": "INCONCLUSIVE",
    }
    consensus = consensus_map.get(consensus_raw, consensus_raw)
    disagreement = bool(
        consensus
        and consensus in {"PASS", "FAIL", "INCONCLUSIVE"}
        and consensus != authoritative
    )

    ref_points = min(ref_count / 20.0, 1.0)
    seed_points = min(seed_count / 3.0, 1.0)
    verifier_points = min(len(pvals) / 3.0, 1.0)
    agreement_points = 0.0 if disagreement else 1.0

    confidence = round(
        0.30 * ref_points
        + 0.20 * seed_points
        + 0.30 * verifier_points
        + 0.20 * agreement_points,
        4,
    )

    factors = [
        {
            "name": "Reference retrains",
            "value": ref_count,
            "status": "STRONG" if ref_count >= 20 else (
                "MODERATE" if ref_count >= 8 else "WEAK"
            ),
        },
        {
            "name": "Independent seeds",
            "value": seed_count,
            "status": "STRONG" if seed_count >= 3 else (
                "MODERATE" if seed_count >= 2 else "WEAK"
            ),
        },
        {
            "name": "Verifier evidence",
            "value": len(pvals),
            "status": "STRONG" if len(pvals) >= 3 else (
                "MODERATE" if pvals else "WEAK"
            ),
        },
        {
            "name": "Verifier agreement",
            "value": not disagreement,
            "status": "STRONG" if not disagreement else "WEAK",
        },
    ]

    quality = (
        "HIGH"
        if ref_count >= 20 and seed_count >= 3 and len(pvals) >= 3 and not disagreement
        else "MODERATE"
        if ref_count >= 8 and len(pvals) >= 1
        else "LOW"
    )
    label = "HIGH" if confidence >= 0.85 else (
        "MODERATE" if confidence >= 0.60 else "LOW"
    )

    return {
        "confidence": confidence,
        "confidence_label": label,
        "evidence_quality": quality,
        "reference_retrains": ref_count,
        "seed_count": seed_count,
        "factors": factors,
        "interpretation": (
            "Confidence is a bounded evidence-strength indicator, not a "
            "calibrated probability of correctness."
        ),
    }
