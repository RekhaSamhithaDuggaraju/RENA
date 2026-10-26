import json
import hashlib
import time
from pathlib import Path
from core.llm_evidence_analyzer import analyze_evidence_constrained, LLMAPIError

INPUT = Path("results/llm_gemini_benchmark/benchmark_v1_1.json")
OUT = Path("results/llm_gemini_benchmark/benchmark_v1_2_constrained_results.json")
CHECKPOINT = Path("results/llm_gemini_benchmark/benchmark_v1_2_constrained_checkpoint.json")

EXPECTED_SHA256 = "a55ee866b8bbabccac8cdc1d2a24dfaf20b404758f1a9b6e74693d9effd46e3c"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def make_evidence(case):
    ev = case["evidence"]
    prov = case["provenance"]

    evidence = []

    witnesses = [
        ("E001", "Bootstrap Statistical Witness"),
        ("E002", "Confidence Analysis"),
        ("E003", "Loss Distribution KS-Test"),
        ("E004", "Membership Inference Attack"),
    ]

    for evidence_id, verifier in witnesses:
        value = ev["witnesses"].get(verifier)

        evidence.append({
            "evidence_id": evidence_id,
            "verifier": verifier,
            "metric": "forget_score",
            "result": value,
            "score": value,
            "reliability": None,
            "supports_forgetting": None,
            "contradictions": [],
            "limitations": [],
            "metadata": {
                "case_id": case["case_id"],
                "method": prov["method"],
                "seed": prov["seed"],
                "forget_fraction": prov["forget_fraction"],
                "evidence_role": "primary_verifier_witness",
            },
        })

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

        evidence.append({
            "evidence_id": evidence_id,
            "verifier": verifier,
            "metric": key,
            "result": value,
            "score": value,
            "reliability": None,
            "supports_forgetting": None,
            "contradictions": [],
            "limitations": (
                [] if value is not None
                else ["Metric unavailable in source experiment"]
            ),
            "metadata": {
                "case_id": case["case_id"],
                "method": prov["method"],
                "seed": prov["seed"],
                "forget_fraction": prov["forget_fraction"],
                "evidence_role": "measured_context",
            },
        })

    return evidence

# ------------------------------------------------------------
# Verify frozen v1.1 input before doing anything
# ------------------------------------------------------------

actual_sha = sha256_file(INPUT)

if actual_sha != EXPECTED_SHA256:
    raise SystemExit(
        f"FROZEN INPUT SHA256 MISMATCH\n"
        f"Expected: {EXPECTED_SHA256}\n"
        f"Actual:   {actual_sha}"
    )

with INPUT.open() as f:
    benchmark = json.load(f)

cases = benchmark["cases"]

if len(cases) != 72:
    raise SystemExit(f"Expected 72 cases, found {len(cases)}")

print("FROZEN INPUT VERIFIED")
print("Cases:", len(cases))
print("SHA256:", actual_sha)
print("MODE: REAL_API_CONSTRAINED")
print()

# ------------------------------------------------------------
# Resume from checkpoint if present
# ------------------------------------------------------------

results = []

if CHECKPOINT.exists():
    with CHECKPOINT.open() as f:
        checkpoint = json.load(f)

    if checkpoint.get("input_sha256") != actual_sha:
        raise SystemExit("Checkpoint belongs to a different frozen input.")

    results = checkpoint.get("results", [])
    print("RESUMING CHECKPOINT:", len(results), "completed cases")
else:
    print("STARTING FRESH")

completed_ids = {r["case_id"] for r in results}

# ------------------------------------------------------------
# Run constrained Gemini audit
# ------------------------------------------------------------

for index, case in enumerate(cases, start=1):

    case_id = case["case_id"]

    if case_id in completed_ids:
        print(f"[{index}/72] {case_id} SKIP - already completed")
        continue

    print(f"[{index}/72] {case_id} RUNNING...", flush=True)

    record = {
        "case_id": case_id,
        "provenance": case["provenance"],
        "reference_verdict": case["provenance"].get("reference_verdict"),
        "reference_score": case["evidence"].get("RENA_FULL_score"),
    }

    try:
        evidence = make_evidence(case)
        analysis = analyze_evidence_constrained(evidence)

        record["status"] = "SUCCESS"
        record["analysis"] = analysis

        print(
            f"    SUCCESS | "
            f"{analysis.get('overall_status')} | "
            f"{analysis.get('claim_strength')} | "
            f"{analysis.get('analysis_mode')}",
            flush=True,
        )

    except LLMAPIError as exc:
        record["status"] = "API_ERROR"
        record["error"] = str(exc)

        print(f"    API_ERROR | {exc}", flush=True)

    except Exception as exc:
        record["status"] = "UNEXPECTED_ERROR"
        record["error"] = repr(exc)

        print(f"    UNEXPECTED_ERROR | {exc}", flush=True)

    results.append(record)

    checkpoint = {
        "benchmark_version": "v1.2_constrained",
        "input_file": str(INPUT),
        "input_sha256": actual_sha,
        "total_cases": 72,
        "results": results,
    }

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    with CHECKPOINT.open("w") as f:
        json.dump(checkpoint, f, indent=2)

    # Small gap between Gemini requests
    if index < len(cases):
        time.sleep(0.5)

# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

output = {
    "benchmark_version": "v1.2_constrained",
    "input_benchmark": "benchmark_v1_1.json",
    "input_sha256": actual_sha,
    "model": "gemini-3.5-flash-lite",
    "analysis_mode": "REAL_API_CONSTRAINED",
    "total_cases": 72,
    "results": results,
}

with OUT.open("w") as f:
    json.dump(output, f, indent=2)

success = [r for r in results if r["status"] == "SUCCESS"]
api_errors = [r for r in results if r["status"] == "API_ERROR"]
unexpected = [r for r in results if r["status"] == "UNEXPECTED_ERROR"]

print()
print("=" * 60)
print("V1.2 CONSTRAINED BENCHMARK COMPLETE")
print("=" * 60)
print("TOTAL:", len(results))
print("SUCCESS:", len(success))
print("API ERRORS:", len(api_errors))
print("UNEXPECTED ERRORS:", len(unexpected))
print("OUTPUT:", OUT)
print("CHECKPOINT:", CHECKPOINT)

from collections import Counter

statuses = Counter()
strengths = Counter()
modes = Counter()

for r in success:
    a = r.get("analysis", {})
    statuses[a.get("overall_status")] += 1
    strengths[a.get("claim_strength")] += 1
    modes[a.get("analysis_mode")] += 1

print()
print("OVERALL STATUS:")
for k, v in sorted(statuses.items()):
    print(f"  {k}: {v}")

print()
print("CLAIM STRENGTH:")
for k, v in sorted(strengths.items()):
    print(f"  {k}: {v}")

print()
print("ANALYSIS MODE:")
for k, v in sorted(modes.items()):
    print(f"  {k}: {v}")
