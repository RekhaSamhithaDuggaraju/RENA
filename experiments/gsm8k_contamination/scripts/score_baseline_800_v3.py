import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path("experiments/gsm8k_contamination")
BASE = ROOT / "data/pilot"
INPUT = BASE / "baseline_qwen_0.5b_100_800tokens.jsonl"
OUTPUT = BASE / "baseline_qwen_0.5b_100_800tokens_scored_auto_v3.jsonl"
SUMMARY = BASE / "baseline_qwen_0.5b_100_800tokens_summary_auto_v3.json"

NUMBER_TEXT = (
    r"[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?"
    r"(?:\s*/\s*\d[\d,]*)?"
)
NUMBER = re.compile(NUMBER_TEXT)

FINAL_MARKER = re.compile(
    r"(?i)\b(?:final\s+answer|the\s+answer\s+is|answer\s+is|"
    r"answer\s*[:=]|final\s*[:=])\s*"
)

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

def boxed_number(text):
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    for content in reversed(boxed):
        matches = NUMBER.findall(content)
        if matches:
            return matches[-1].strip()
    return None

def extract_prediction(response):
    text = response or ""

    # Prefer the final explicit answer marker, not earlier reasoning.
    markers = list(FINAL_MARKER.finditer(text))
    if markers:
        marker = markers[-1]
        tail = text[marker.end():marker.end() + 250]

        # Common format: Final answer is \boxed{20}.
        boxed = boxed_number(tail)
        if boxed is not None:
            return boxed, "boxed_after_final_marker_v3"

        # Common format: Final answer: 20 cookies.
        immediate = NUMBER.match(tail.strip())
        if immediate:
            return immediate.group(0).strip(), "explicit_final_number_v3"

        # Common format: Final answer: Each person paid $20.
        # Search only a short window after the marker.
        nearby = NUMBER.search(tail[:120])
        if nearby:
            return nearby.group(0).strip(), "number_near_final_marker_v3"

        # Avoid substituting an unrelated number from earlier reasoning.
        return None, "non_numeric_final_answer_needs_review"

    # If no explicit marker exists, try the last boxed numeric answer.
    boxed = boxed_number(text)
    if boxed is not None:
        return boxed, "boxed_answer_v3"

    # Last resort: final numeric expression, always manually reviewable.
    matches = NUMBER.findall(text[-500:])
    if matches:
        return matches[-1].strip(), "last_number_fallback_v3"

    return None, "no_numeric_answer_needs_review"

if not INPUT.exists():
    raise SystemExit(f"ERROR: Missing input: {INPUT}")

if OUTPUT.exists() or SUMMARY.exists():
    raise SystemExit(
        "STOP: A v3 output already exists; nothing overwritten.\n"
        f"{OUTPUT}\n{SUMMARY}"
    )

with INPUT.open() as f:
    rows = [json.loads(line) for line in f if line.strip()]

if len(rows) != 100:
    raise SystemExit(f"ERROR: Expected 100 rows, found {len(rows)}")

ids = [r.get("example_id") for r in rows]
if len(set(ids)) != len(ids):
    raise SystemExit("ERROR: Duplicate example IDs.")

scored = []
statuses = Counter()
methods = Counter()

for row in rows:
    prediction, method = extract_prediction(row.get("model_response", ""))
    reference = normalize_number(str(row.get("reference_final_answer", "")))
    predicted = normalize_number(prediction)

    if row.get("token_limit_reached", False):
        status, correct = "unscorable_token_limit", None
    elif predicted is None or reference is None:
        status, correct = "needs_manual_review", None
    else:
        correct = predicted == reference
        status = "correct" if correct else "incorrect"

    result = dict(row)
    result.update({
        "predicted_final_answer_auto": prediction,
        "reference_numeric_normalized": (
            str(reference) if reference is not None else None
        ),
        "predicted_numeric_normalized": (
            str(predicted) if predicted is not None else None
        ),
        "auto_correct": correct,
        "auto_score_status": status,
        "auto_parse_method": method,
        "needs_manual_review": (
            status in {"unscorable_token_limit", "needs_manual_review"}
            or method == "last_number_fallback_v3"
        ),
        "scorer_version": "auto_v3",
    })
    scored.append(result)
    statuses[status] += 1
    methods[method] += 1

with OUTPUT.open("w") as f:
    for row in scored:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

scorable = statuses["correct"] + statuses["incorrect"]
summary = {
    "scorer_version": "auto_v3",
    "input_file": str(INPUT),
    "output_file": str(OUTPUT),
    "total_rows": len(rows),
    "correct_auto": statuses["correct"],
    "incorrect_auto": statuses["incorrect"],
    "needs_manual_review": statuses["needs_manual_review"],
    "unscorable_token_limit": statuses["unscorable_token_limit"],
    "scorable_rows": scorable,
    "accuracy_among_automatically_scorable": (
        round(statuses["correct"] / scorable, 4) if scorable else None
    ),
    "accuracy_over_all_100_using_auto_correct_only": round(
        statuses["correct"] / 100, 4
    ),
    "parse_methods": dict(methods),
    "caveat": (
        "Provisional numeric-match scores only. Review fallback and ambiguous "
        "outputs manually; numeric agreement does not validate reasoning."
    ),
}

with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("SCORING COMPLETE — AUTO V3")
print(json.dumps(summary, indent=2))

by_id = {r["example_id"]: r for r in scored}
print("\nCHECK SELECTED ROWS")
for eid in (
    "gsm8k_test_0000", "gsm8k_test_0004", "gsm8k_test_0005",
    "gsm8k_test_0012", "gsm8k_test_0039", "gsm8k_test_0045",
    "gsm8k_test_0064", "gsm8k_test_0086", "gsm8k_test_0099"
):
    r = by_id[eid]
    print(
        eid,
        "| ref:", r.get("reference_final_answer"),
        "| predicted:", r.get("predicted_final_answer_auto"),
        "| status:", r.get("auto_score_status"),
        "| method:", r.get("auto_parse_method"),
    )

print("\nCreated:")
print(OUTPUT)
print(SUMMARY)
