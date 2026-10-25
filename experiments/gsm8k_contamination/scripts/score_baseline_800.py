import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from collections import Counter

ROOT = Path("experiments/gsm8k_contamination")
DATA = ROOT / "data/pilot"

INPUT = DATA / "baseline_qwen_0.5b_100_800tokens.jsonl"
OUTPUT = DATA / "baseline_qwen_0.5b_100_800tokens_scored_auto_v1.jsonl"
SUMMARY = DATA / "baseline_qwen_0.5b_100_800tokens_summary_auto_v1.json"

NUMBER = re.compile(
    r"(?<![\w.])[-+]?\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*)?(?![\w.])"
)

def normalize_number(value):
    """Normalize a simple integer, decimal, currency amount, or fraction."""
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
    """Prefer boxed answers, then explicit final-answer statements."""
    text = response or ""

    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        matches = NUMBER.findall(boxed[-1])
        if matches:
            return matches[-1].strip(), "boxed_answer"

    # Look for the last explicit final-answer marker.
    markers = list(re.finditer(
        r"(?i)\b(?:final answer|answer is|answer:|answer\s*=)\s*",
        text
    ))
    if markers:
        tail = text[markers[-1].end():markers[-1].end() + 300]
        matches = NUMBER.findall(tail)
        if matches:
            return matches[0].strip(), "explicit_answer_marker"

    # Conservative fallback: inspect only the end of the response.
    tail = text[-500:]
    matches = NUMBER.findall(tail)
    if matches:
        return matches[-1].strip(), "last_number_fallback"

    return None, "no_numeric_answer_found"

if not INPUT.exists():
    raise SystemExit(f"ERROR: Missing input file: {INPUT}")

if OUTPUT.exists() or SUMMARY.exists():
    raise SystemExit(
        "ERROR: A target output already exists. Nothing was overwritten.\n"
        f"Output: {OUTPUT}\nSummary: {SUMMARY}"
    )

rows = []
with INPUT.open() as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

if len(rows) != 100:
    raise SystemExit(f"ERROR: Expected 100 rows, found {len(rows)}")

ids = [row.get("example_id") for row in rows]
if len(set(ids)) != len(ids):
    raise SystemExit("ERROR: Duplicate example IDs detected.")

scored = []
counts = Counter()
parse_methods = Counter()

for row in rows:
    prediction, parse_method = extract_prediction(row.get("model_response", ""))
    reference = normalize_number(str(row.get("reference_final_answer", "")))
    predicted_value = normalize_number(prediction) if prediction is not None else None

    # Truncated generations are not automatically scored, even if a number
    # can be extracted from their incomplete reasoning.
    if row.get("token_limit_reached", False):
        status = "unscorable_token_limit"
        correct = None
    elif predicted_value is None or reference is None:
        status = "needs_manual_review"
        correct = None
    else:
        correct = predicted_value == reference
        status = "correct" if correct else "incorrect"

    result = dict(row)
    result.update({
        "predicted_final_answer_auto": prediction,
        "reference_numeric_normalized": str(reference) if reference is not None else None,
        "predicted_numeric_normalized": str(predicted_value) if predicted_value is not None else None,
        "auto_correct": correct,
        "auto_score_status": status,
        "auto_parse_method": parse_method,
        "needs_manual_review": status in {
            "unscorable_token_limit",
            "needs_manual_review",
        } or parse_method == "last_number_fallback",
        "scorer_version": "auto_v1",
    })

    scored.append(result)
    counts[status] += 1
    parse_methods[parse_method] += 1

with OUTPUT.open("w") as f:
    for row in scored:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

scorable = counts["correct"] + counts["incorrect"]
correct = counts["correct"]

summary = {
    "scorer_version": "auto_v1",
    "input_file": str(INPUT),
    "output_file": str(OUTPUT),
    "total_rows": len(rows),
    "unique_example_ids": len(set(ids)),
    "correct_auto": correct,
    "incorrect_auto": counts["incorrect"],
    "unscorable_token_limit": counts["unscorable_token_limit"],
    "needs_manual_review_unparsed": counts["needs_manual_review"],
    "scorable_rows": scorable,
    "accuracy_among_automatically_scorable": (
        round(correct / scorable, 4) if scorable else None
    ),
    "accuracy_over_all_100_using_only_confirmed_auto_scores": round(correct / 100, 4),
    "parse_methods": dict(parse_methods),
    "important_caveat": (
        "Provisional automated numeric scoring only. Fallback extractions and "
        "all token-limited responses require review. Numeric agreement does not "
        "guarantee valid reasoning; manually validate before reporting final accuracy."
    ),
}

with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("SCORING COMPLETE")
print("Total rows:", len(rows))
print("Automatically correct:", counts["correct"])
print("Automatically incorrect:", counts["incorrect"])
print("Token-limited / unscorable:", counts["unscorable_token_limit"])
print("Other unscorable:", counts["needs_manual_review"])
print("Automatically scorable:", scorable)
print("Accuracy among automatically scorable:",
      summary["accuracy_among_automatically_scorable"])
print("\nParse methods:")
for method, count in sorted(parse_methods.items()):
    print(f"  {method}: {count}")
print("\nCreated:")
print(OUTPUT)
print(SUMMARY)
print("\nReminder: These are provisional automatic scores, not manually validated results.")
