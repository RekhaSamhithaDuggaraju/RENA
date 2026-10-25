"""Paired NIAH comparison against a retain-only reference checkpoint.

The comparison uses identical case IDs and fact-cluster bootstrap resampling;
it reports behavioral gaps, not proof of parameter erasure.
"""
from __future__ import annotations
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _load_rows(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: expected non-empty cases list")
    result = {}
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}: case {i} must be an object")
        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"{label}: case {i} has invalid case_id")
        if cid in result:
            raise ValueError(f"{label}: duplicate case_id {cid}")
        if not isinstance(row.get("retrieved"), bool):
            raise ValueError(f"{label}: {cid} retrieved must be boolean")
        if row.get("target_split") not in {"forget", "retain"}:
            raise ValueError(f"{label}: {cid} target_split invalid")
        if row.get("context_mode") not in {"present", "absent"}:
            raise ValueError(f"{label}: {cid} context_mode invalid")
        result[cid] = dict(row)
    return result


def _rate(rows: Sequence[Mapping[str, Any]], split: str, mode: str) -> float | None:
    selected = [r for r in rows if r["target_split"] == split and r["context_mode"] == mode]
    return sum(r["retrieved"] for r in selected) / len(selected) if selected else None


def _cluster_bootstrap(deltas: list[tuple[str, float]], *, seed: int, iterations: int) -> list[float] | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for cluster, value in deltas:
        groups[cluster].append(value)
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        chosen = rng.choices(keys, k=len(keys))
        vals = [v for key in chosen for v in groups[key]]
        samples.append(sum(vals) / len(vals))
    samples.sort()
    return [samples[int(.025 * (len(samples)-1))], samples[int(.975 * (len(samples)-1))]]


def compare_niah_to_reference(candidate_payload: Mapping[str, Any], reference_payload: Mapping[str, Any], *,
                              candidate_name: str = "candidate", reference_name: str = "retain_only_retraining",
                              seed: int = 42, bootstrap_iterations: int = 2000) -> dict[str, Any]:
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be >= 100")
    candidate = _load_rows(candidate_payload, candidate_name)
    reference = _load_rows(reference_payload, reference_name)
    shared = sorted(set(candidate) & set(reference))
    if not shared:
        raise ValueError("candidate and reference have no matched case IDs")
    mismatches = []
    paired = []
    for cid in shared:
        a, b = candidate[cid], reference[cid]
        fields = ("fact_id", "target_split", "context_mode", "insertion_position", "prompt_style", "context_length_chars")
        if any(a.get(f) != b.get(f) for f in fields):
            mismatches.append(cid)
        else:
            paired.append((a, b))
    if mismatches:
        raise ValueError(f"matched case metadata differs for {mismatches[:5]}")
    if not paired:
        raise ValueError("no metadata-matched cases")
    arows = [a for a, _ in paired]
    brows = [b for _, b in paired]
    def delta(split: str, mode: str) -> dict[str, Any]:
        pairs = [(a, b) for a, b in paired if a["target_split"] == split and a["context_mode"] == mode]
        if not pairs:
            return {"n": 0, "candidate_rate": None, "reference_rate": None, "difference": None, "cluster_bootstrap_95ci": None}
        vals = [(str(a.get("fact_id", a["case_id"])), float(a["retrieved"]) - float(b["retrieved"])) for a,b in pairs]
        diff = sum(v for _,v in vals) / len(vals)
        return {"n": len(vals), "candidate_rate": sum(a["retrieved"] for a,_ in pairs)/len(pairs),
                "reference_rate": sum(b["retrieved"] for _,b in pairs)/len(pairs), "difference": diff,
                "cluster_bootstrap_95ci": _cluster_bootstrap(vals, seed=seed, iterations=bootstrap_iterations)}
    return {"analysis": "RENA-paired-NIAH-reference-v1", "candidate": candidate_name,
            "reference": reference_name, "seed": seed, "matched_case_count": len(paired),
            "candidate_only_case_count": len(set(candidate)-set(reference)),
            "reference_only_case_count": len(set(reference)-set(candidate)),
            "forget_absent_disclosure": delta("forget", "absent"),
            "forget_present_retrieval": delta("forget", "present"),
            "retain_present_retrieval": delta("retain", "present"),
            "interpretation": {"difference": "candidate retrieval rate minus reference retrieval rate; positive means more retrieval by candidate",
              "caution": "Behavioral comparison only; synthetic facts and exact-match retrieval do not prove parameter erasure.",
              "interval": "Fact-cluster bootstrap; descriptive and sensitive to the number of unique facts."}}
