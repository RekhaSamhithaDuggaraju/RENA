"""Traceable, serializable evidence derived from RENA verification outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvidenceItem:
    evidence_id: str
    verifier: str
    metric: str
    result: float | str
    score: float | None
    reliability: float | None
    supports_forgetting: bool | None
    contradictions: list[str]
    limitations: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def detect_contradictions(items: list[EvidenceItem], threshold: float = 0.5) -> list[EvidenceItem]:
    """Annotate low forgetting-score verifiers when peer evidence supports forgetting."""
    supports = [x for x in items if x.score is not None and x.score >= threshold]
    conflicts = [x.evidence_id for x in items if x.score is not None and x.score < threshold]
    if supports and conflicts:
        for item in items:
            if item.evidence_id in conflicts:
                item.contradictions = [x.evidence_id for x in supports]
            elif item.score is not None and item.score >= threshold:
                item.contradictions = conflicts
    return items


def evidence_from_audit(algorithm: str, audit: dict, start_index: int = 1) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for offset, module in enumerate(audit["module_results"]):
        score = float(module["forget_score"])
        items.append(EvidenceItem(
            evidence_id=f"E{start_index + offset:03d}", verifier=module["module_name"],
            metric="forget_score", result=float(module["raw_statistic"]), score=score,
            reliability=float(module["reliability"]), supports_forgetting=score >= 0.5,
            contradictions=[], limitations=[module["reliability_reason"]],
            metadata={"algorithm": algorithm, "p_value": module["p_value"], **module["detail"]},
        ))
    return detect_contradictions(items)
