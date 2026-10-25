import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path("experiments/gsm8k_contamination")
BASE = ROOT / "data/pilot"
INPUT = BASE / "baseline_qwen_0.5b_100_800tokens.jsonl"
OUTPUT = BASE / "baseline_qwen_0.5b_100_800tokens_scored_auto_v4.jsonl"
SUMMARY = BASE / "baseline_qwen_0.5b_100_800tokens_summary_auto_v4.json"

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

def boxed_numbers(text):
    contents = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    result = []
    for content in contents:
        matches = NUMBER.findall(content)
        if matches:
            result.append(matches[-1].strip())
    return result

def extract_prediction(response):
    text = response or ""

    # Work from the last explicit answer marker.
    markers = list(FINAL_MARKER.finditer(text))
    if markers:
        marker = markers[-1]
        tail = text[marker.end():marker.end() + 180].strip()

        # Prefer an answer explicitly boxed after the final marker.
        boxed = boxed_numbers(tail)
        if boxed:
            return boxed[-1], "boxed_after_final_marker_v4"

        # Accept a number immediately after the marker.
        immediate = NUMBER.match(tail)
        if immediate:
            return immediate.group(0).strip(), "immediate_final_number_v4"

        # Only accept a nearby number if the intervening text is short
        # and looks like ordinary answer wording, not a new calculation.
        nearby = NUMBER.search(tail[:70])
        if nearby:
            prefix = tail[:nearby.start()].strip()
            if (
                len(prefix) <= 45
                and not re.search(r"[=+\*/]", prefix)
                and not re.search(r"\d", prefix)
            ):
                return nearby.group(0).strip(), "number_near_final_marker_v4"

        return None, "final_marker_ambiguous_review"

    # No explicit marker: a single boxed answer is relatively strong evidence.
    boxed = boxed_numbers(text)
    if boxed:
        return boxed[-1], "boxed_answer_v4"

    # Without a final marker or boxed answer, don't guess from a chain of
    # calculations. Such responses need manual review.
    return None, "no_reliable_final_answer_review"

if not INPUT.exists():
    raise SystemExit(f"ERROR: Missing input: {INPUT}")

if OUTPUT.exists() or SUMMARY.exists():
    raise SystemExit(
        "STOP: A v4 output already exists; nothing overwritten.\n"
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
        ),
        "scorer_version": "auto_v4",
    })
    scored.append(result)
    statuses[status] += 1
    methods[method] += 1

with OUTPUT.open("w") as f:
    for row in scored:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

scorable = statuses["correct"] + statuses["incorrect"]
summary = {
    "scorer_version": "auto_v4",
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
        "Provisional numeric-match scores only. Unit equivalence, reference "
        "quality, and reasoning correctness require manual validation."
    ),
}

with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("SCORING COMPLETE — AUTO V4")
print(json.dumps(summary, indent=2))

by_id = {r["example_id"]: r for r in scored}
print("\nCHECK SELECTED ROWS")
for eid in (
    "gsm8k_test_0001", "gsm8k_test_0012", "gsm8k_test_0013",
    "gsm8k_test_0027", "gsm8k_test_0054", "gsm8k_test_0056",
    "gsm8k_test_0060", "gsm8k_test_0081", "gsm8k_test_0099"
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
