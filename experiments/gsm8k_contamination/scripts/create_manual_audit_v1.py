import json
from pathlib import Path
from collections import Counter

BASE = Path("experiments/gsm8k_contamination")
INPUT = BASE / "data/pilot/baseline_qwen_0.5b_100_800tokens_scored_auto_v4.jsonl"
OUTPUT = BASE / "data/pilot/baseline_qwen_0.5b_100_800tokens_manual_audit_v1.jsonl"
SUMMARY = BASE / "data/pilot/baseline_qwen_0.5b_100_800tokens_manual_audit_summary_v1.json"

# Provisional decisions based on the response-tail review.
# These do not overwrite or replace the original automatic scores.
decisions = {
    "0002": ("provisional_correct", "84 cents", "Final answer matches reference."),
    "0003": ("provisional_correct", "$14,400", "Final answer matches reference."),
    "0009": ("provisional_incorrect", "330", "Final answer differs from reference 370."),
    "0012": ("needs_reference_check", "No additional chairs are needed", "Supplied reference is 9, but the response says no additional chairs are needed; verify the question and reference."),
    "0013": ("provisional_incorrect", "58 yards", "Final answer differs from reference 16."),
    "0014": ("provisional_incorrect", "390 seconds", "Final answer differs from reference 540."),
    "0018": ("provisional_incorrect", "$11.25", "Final answer differs from reference 9."),
    "0019": ("provisional_correct", "16 pounds", "Numeric answer matches reference 16."),
    "0020": ("provisional_correct", "120 miles", "Numeric answer matches reference 120."),
    "0024": ("provisional_incorrect", "Approximately 5.33 hours", "Response duration differs from reference; the stated grouping calculation should be checked."),
    "0025": ("provisional_incorrect", "$456", "Final answer differs from reference 60."),
    "0027": ("provisional_incorrect", "60", "Final answer differs from reference 98."),
    "0028": ("provisional_incorrect", "900 miles", "Final answer differs from reference 300."),
    "0029": ("provisional_incorrect", "25 hours", "Final answer differs from reference 48."),
    "0030": ("provisional_correct", "7 packs", "Numeric answer matches reference 7."),
    "0031": ("provisional_correct", "200 minutes", "Numeric answer matches reference 200."),
    "0034": ("provisional_correct", "623 lb", "Numeric answer matches reference 623."),
    "0039": ("provisional_correct", "-3 degrees C", "Numeric answer matches reference -3."),
    "0040": ("provisional_incorrect", "$1.70", "Final answer differs from reference 15."),
    "0047": ("provisional_incorrect", "$10,800", "Final answer differs from reference 64,800."),
    "0054": ("provisional_incorrect", "52 cupcakes", "Final answer differs from reference 54."),
    "0056": ("provisional_correct", "15 gallons", "Final answer matches reference 15; earlier automatic extraction selected an intermediate value."),
    "0057": ("provisional_incorrect", "$12,000", "Final answer differs from reference 3,000."),
    "0060": ("provisional_correct", "20 dozens", "Numeric answer matches reference 20."),
    "0061": ("provisional_correct", "$21", "Numeric answer matches reference 21."),
    "0062": ("provisional_incorrect", "15 weeks", "Final answer differs from reference 7."),
    "0064": ("provisional_incorrect", "20 containers", "Final answer differs from reference 4."),
    "0068": ("provisional_incorrect", "$68", "Final answer differs from reference 6."),
    "0070": ("provisional_correct", "15 minutes", "Numeric answer matches reference 15."),
    "0074": ("provisional_incorrect", "$495", "Final answer differs from reference 255."),
    "0078": ("provisional_incorrect", "6", "Final answer differs from reference 70."),
    "0079": ("provisional_correct", "$4", "Numeric answer matches reference 4."),
    "0080": ("provisional_correct", "2 friends", "Numeric answer matches reference 2."),
    "0081": ("provisional_incorrect", "$24", "Final answer differs from reference 64."),
    "0083": ("provisional_correct", "17,000", "Numeric answer matches reference 17,000."),
}

if not INPUT.exists():
    raise SystemExit(f"Missing input file: {INPUT}")

with INPUT.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if len(rows) != 100:
    raise SystemExit(f"Expected 100 baseline rows, found {len(rows)}. No output written.")

found_ids = {
    str(row.get("example_id", "")).replace("gsm8k_test_", "")
    for row in rows
}
missing = sorted(set(decisions) - found_ids)
if missing:
    raise SystemExit(f"These audit IDs were not found in the input: {missing}. No output written.")

audited = []
for row in rows:
    item = dict(row)
    short_id = str(item.get("example_id", "")).replace("gsm8k_test_", "")

    if short_id in decisions:
        status, answer, reason = decisions[short_id]
        item["manual_audit_status"] = status
        item["manual_audit_answer"] = answer
        item["manual_audit_reason"] = reason
        item["manual_audit_version"] = "v1"
        item["manual_audit_basis"] = (
            "Provisional response-tail review; reference answers have not all been independently verified."
        )
    else:
        item["manual_audit_status"] = "not_reviewed_in_this_pass"
        item["manual_audit_answer"] = None
        item["manual_audit_reason"] = None
        item["manual_audit_version"] = "v1"

    audited.append(item)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8") as f:
    for row in audited:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

counts = Counter(row["manual_audit_status"] for row in audited)
summary = {
    "audit_version": "v1",
    "input_file": str(INPUT),
    "output_file": str(OUTPUT),
    "total_baseline_rows": len(rows),
    "rows_decided_provisionally": len(decisions),
    "provisional_correct": counts["provisional_correct"],
    "provisional_incorrect": counts["provisional_incorrect"],
    "needs_reference_check": counts["needs_reference_check"],
    "not_reviewed_in_this_pass": counts["not_reviewed_in_this_pass"],
    "important_note": (
        "These are provisional manual decisions for the previously flagged rows, "
        "not final verified baseline accuracy. Check references and audit the "
        "automatically scored rows before reporting accuracy."
    ),
}
with SUMMARY.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(json.dumps(summary, indent=2))
print("\nCreated:")
print(OUTPUT)
print(SUMMARY)
print("\nOriginal V4 files were not modified.")
