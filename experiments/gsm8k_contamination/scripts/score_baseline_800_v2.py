import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path("experiments/gsm8k_contamination")
BASE = ROOT / "data/pilot"
INPUT = BASE / "baseline_qwen_0.5b_100_800tokens.jsonl"
OUTPUT = BASE / "baseline_qwen_0.5b_100_800tokens_scored_auto_v2.jsonl"
SUMMARY = BASE / "baseline_qwen_0.5b_100_800tokens_summary_auto_v2.json"

NUMBER_TEXT = (
    r"[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?"
    r"(?:\s*/\s*\d[\d,]*)?"
)

FINAL_MARKER = re.compile(
    r"(?i)\b(?:final\s+answer|the\s+answer\s+is|answer\s+is|"
    r"answer\s*[:=]|final\s*[:=])\s*"
)

IMMEDIATE_NUMBER = re.compile(
    r"^\s*(?:\*\*)?(?:\\\()?\s*(" + NUMBER_TEXT + r")",
    re.I,
)

NUMBER = re.compile(NUMBER_TEXT)


def normalize_number(value):
    if value is None:
        return None
    value = value.strip().replace("$", "").replace(",", "")
    value = re.sub(r"\s+", "", value)
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            result = Decimal(numerator) / Decimal(denominator)
        else:
            result = Decimal(value)
        return result.normalize()
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def extract_prediction(response):
    text = response or ""

    # Prefer the last explicit answer statement. If it is nonnumeric,
    # do not silently substitute a number from earlier reasoning.
    markers = list(FINAL_MARKER.finditer(text))
    if markers:
        marker = markers[-1]
        tail = text[marker.end():marker.end() + 100]
        match = IMMEDIATE_NUMBER.match(tail)
        if match:
            return match.group(1).strip(), "explicit_final_answer_v2"
        return None, "non_numeric_final_answer_needs_review"

    # Next, use the last boxed numeric answer.
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        matches = NUMBER.findall(boxed[-1])
        if matches:
            return matches[-1].strip(), "boxed_answer_v2"

    # Last resort: use the final numeric expression in the response.
    # This is always marked for manual review.
    tail = text[-500:]
    matches = NUMBER.findall(tail)
    if matches:
        return matches[-1].strip(), "last_number_fallback_v2"

    return None, "no_numeric_answer_needs_review"


if not INPUT.exists():
    raise SystemExit(f"ERROR: Missing input: {INPUT}")

if OUTPUT.exists() or SUMMARY.exists():
    raise SystemExit(
        "STOP: A v2 output already exists; nothing overwritten.\n"
        f"{OUTPUT}\n{SUMMARY}"
    )

rows = []
with INPUT.open() as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

if len(rows) != 100:
    raise SystemExit(f"ERROR: Expected 100 rows, found {len(rows)}")

ids = [r.get("example_id") for r in rows]
if len(set(ids)) != len(ids):
    raise SystemExit("ERROR: Duplicate example IDs.")

scored = []
counts = Counter()
methods = Counter()

for row in rows:
    prediction, method = extract_prediction(row.get("model_response", ""))
    reference = normalize_number(str(row.get("reference_final_answer", "")))
    predicted = normalize_number(prediction)

    if row.get("token_limit_reached", False):
        status, correct = "unscorable_token_limit", None
    elif method == "non_numeric_final_answer_needs_review":
        status, correct = "needs_manual_review", None
    elif predicted is None or reference is None:
        status, correct = "needs_manual_review", None
    else:
        correct = predicted == reference
        status = "correct" if correct else "incorrect"

    result = dict(row)
    result.update({
        "predicted_final_answer_auto": prediction,
        "reference_numeric_normalized": str(reference) if reference is not None else None,
        "predicted_numeric_normalized": str(predicted) if predicted is not None else None,
        "auto_correct": correct,
        "auto_score_status": status,
        "auto_parse_method": method,
        "needs_manual_review": (
            status in {"unscorable_token_limit", "needs_manual_review"}
            or method == "last_number_fallback_v2"
        ),
        "scorer_version": "auto_v2",
    })
    scored.append(result)
    counts[status] += 1
    methods[method] += 1

with OUTPUT.open("w") as f:
    for row in scored:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

scorable = counts["correct"] + counts["incorrect"]
summary = {
    "scorer_version": "auto_v2",
    "input_file": str(INPUT),
    "output_file": str(OUTPUT),
    "total_rows": len(rows),
    "correct_auto": counts["correct"],
    "incorrect_auto": counts["incorrect"],
    "needs_manual_review": counts["needs_manual_review"],
    "unscorable_token_limit": counts["unscorable_token_limit"],
    "scorable_rows": scorable,
    "accuracy_among_automatically_scorable": (
        round(counts["correct"] / scorable, 4) if scorable else None
    ),
    "accuracy_over_all_100_using_auto_correct_only": round(
        counts["correct"] / 100, 4
    ),
    "parse_methods": dict(methods),
    "caveat": (
        "Provisional automated numeric scores, not final benchmark accuracy. "
        "Fallbacks and ambiguous responses require manual review. Numeric "
        "agreement alone does not validate reasoning."
    ),
}

with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("SCORING COMPLETE — AUTO V2")
print(json.dumps(summary, indent=2))

by_id = {r["example_id"]: r for r in scored}
for eid in ("gsm8k_test_0001", "gsm8k_test_0002", "gsm8k_test_0012",
            "gsm8k_test_0099"):
    r = by_id[eid]
    print(
        eid,
        "| reference:", r.get("reference_final_answer"),
        "| extracted:", r.get("predicted_final_answer_auto"),
        "| method:", r.get("auto_parse_method"),
        "| correct:", r.get("auto_correct"),
        "| status:", r.get("auto_score_status"),
    )

print("\nCreated:")
print(OUTPUT)
print(SUMMARY)
