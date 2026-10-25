from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

from core.llm_evidence_analyzer import LLMAPIError, analyze_evidence


BENCHMARK = Path("results/llm_gemini_benchmark/benchmark_v1_1.json")
OUTPUT = Path("results/llm_gemini_benchmark/benchmark_v1_1_gemini_results.json")
CHECKPOINT = Path("results/llm_gemini_benchmark/benchmark_v1_1_gemini_checkpoint.json")

MODEL = "gemini-3.5-flash-lite"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_evidence(case: dict) -> list[dict]:
    ev = case["evidence"]
    provenance = case["provenance"]

    witness_names = [
        ("E001", "Bootstrap Statistical Witness", "bootstrap_score"),
        ("E002", "Confidence Analysis", "confidence_score"),
        ("E003", "Loss Distribution KS-Test", "ks_score"),
        ("E004", "Membership Inference Attack", "membership_score"),
    ]

    evidence = []

    for evidence_id, verifier, metric in witness_names:
        value = ev["witnesses"].get(verifier)
        evidence.append({
            "evidence_id": evidence_id,
            "verifier": verifier,
            "metric": metric,
            "result": value,
            "score": value,
            "reliability": None,
            "supports_forgetting": None,
            "contradictions": [],
            "limitations": [],
            "metadata": {
                "case_id": case["case_id"],
                "method": provenance["method"],
                "seed": provenance["seed"],
                "forget_fraction": provenance["forget_fraction"],
                "evidence_role": "primary_verifier_witness",
            },
        })

    # Additional measured context. These are deliberately unlabeled with
    # respect to support/contradiction so Gemini must interpret them.
    context_metrics = [
        ("E005", "Forget Accuracy", "forget_accuracy"),
        ("E006", "Retain Accuracy", "retain_accuracy"),
        ("E007", "Test Accuracy", "test_accuracy"),
        ("E008", "Relative Representation Accuracy", "rra"),
        ("E009", "Parameter Relative Representation Accuracy", "parameter_rra"),
        ("E010", "Prediction Disagreement Rate", "prediction_disagreement_rate"),
        ("E011", "Runtime", "runtime_seconds"),
    ]

    for evidence_id, verifier, key in context_metrics:
        value = ev.get(key)

        # Preserve missing measurements as explicit unavailable evidence.
        limitations = []
        if value is None:
            limitations.append("Metric unavailable in source experiment")

        evidence.append({
            "evidence_id": evidence_id,
            "verifier": verifier,
            "metric": key,
            "result": value,
            "score": value,
            "reliability": None,
            "supports_forgetting": None,
            "contradictions": [],
            "limitations": limitations,
            "metadata": {
                "case_id": case["case_id"],
                "method": provenance["method"],
                "seed": provenance["seed"],
                "forget_fraction": provenance["forget_fraction"],
                "evidence_role": "measured_context",
            },
        })

    return evidence


def load_cases() -> dict:
    with BENCHMARK.open() as f:
        return json.load(f)


def load_checkpoint() -> dict:
    if not CHECKPOINT.exists():
        return {"results": []}

    with CHECKPOINT.open() as f:
        return json.load(f)


def save_checkpoint(payload: dict) -> None:
    tmp = CHECKPOINT.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(CHECKPOINT)


def main() -> None:
    benchmark = load_cases()
    cases = benchmark["cases"]

    benchmark_sha256 = sha256_file(BENCHMARK)

    checkpoint = load_checkpoint()
    completed = {
        item["case_id"]: item
        for item in checkpoint.get("results", [])
    }

    print("RENA Gemini Benchmark v1.1")
    print("Benchmark cases:", len(cases))
    print("Benchmark SHA256:", benchmark_sha256)
    print("Model:", MODEL)
    print("Already completed:", len(completed))
    print()

    for index, case in enumerate(cases, start=1):
        case_id = case["case_id"]

        if case_id in completed:
            print(f"[{index}/{len(cases)}] {case_id} SKIP (checkpoint)")
            continue

        print(
            f"[{index}/{len(cases)}] {case_id} "
            f"method={case['provenance']['method']} "
            f"seed={case['provenance']['seed']} "
            f"forget={case['provenance']['forget_fraction']}"
        )

        evidence = make_evidence(case)

        record = {
            "case_id": case_id,
            "index": index,
            "provenance": case["provenance"],
            # Reference verdict is retained for later evaluation only.
            # It is NOT included in the Gemini evidence packet.
            "reference_verdict": case["provenance"].get("reference_verdict"),
            "reference_score": case["provenance"].get("reference_score"),
            "evidence_sent_to_gemini": evidence,
            "model": MODEL,
        }

        try:
            analysis = analyze_evidence(evidence, model=MODEL)
            record["status"] = "SUCCESS"
            record["analysis"] = analysis

            print(
                f"    -> {analysis['analysis_mode']} "
                f"{analysis['overall_status']} "
                f"{analysis['claim_strength']}"
            )

        except LLMAPIError as exc:
            record["status"] = "API_ERROR"
            record["error"] = str(exc)

            print("    -> API_ERROR:", exc)

        except Exception as exc:
            record["status"] = "UNEXPECTED_ERROR"
            record["error"] = f"{type(exc).__name__}: {exc}"

            print("    -> UNEXPECTED_ERROR:", exc)

        completed[case_id] = record

        # Checkpoint after EVERY case.
        save_checkpoint({
            "benchmark": str(BENCHMARK),
            "benchmark_sha256": benchmark_sha256,
            "model": MODEL,
            "total_cases": len(cases),
            "results": [
                completed[c["case_id"]]
                for c in cases
                if c["case_id"] in completed
            ],
        })

        # Small pacing gap between API calls.
        time.sleep(0.5)

    ordered_results = [
        completed[c["case_id"]]
        for c in cases
        if c["case_id"] in completed
    ]

    output = {
        "benchmark": {
            "source": str(BENCHMARK),
            "sha256": benchmark_sha256,
            "selection_frozen_before_gemini": True,
            "total_cases": len(cases),
        },
        "provider": "Google Gemini",
        "model": MODEL,
        "analysis_policy": {
            "reference_verdict_held_out": True,
            "reference_score_held_out": True,
            "rena_full_verdict_held_out": True,
            "rena_full_score_held_out": True,
            "numerical_claim_guard_enabled": True,
            "checkpoint_after_each_case": True,
        },
        "results": ordered_results,
    }

    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(output, f, indent=2)
    tmp.replace(OUTPUT)

    success = sum(r["status"] == "SUCCESS" for r in ordered_results)
    api_errors = sum(r["status"] == "API_ERROR" for r in ordered_results)
    unexpected = sum(r["status"] == "UNEXPECTED_ERROR" for r in ordered_results)

    print()
    print("========================================")
    print("BENCHMARK COMPLETE")
    print("========================================")
    print("Total:", len(ordered_results))
    print("Successful:", success)
    print("API errors:", api_errors)
    print("Unexpected errors:", unexpected)
    print("Output:", OUTPUT)
    print("Checkpoint:", CHECKPOINT)


if __name__ == "__main__":
    main()
