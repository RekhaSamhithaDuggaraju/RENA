import json
from collections import Counter
from pathlib import Path

base = Path("experiments/gsm8k_contamination/data/pilot")
source = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_final_v1.jsonl"
output = base / "baseline_qwen_0.5b_100_800tokens_baseline_summary_v1.json"

overrides = {
    "gsm8k_test_0012": "incorrect",
    "gsm8k_test_0022": "incorrect",
    "gsm8k_test_0094": "incorrect",
}

rows = []
with source.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

counts = Counter()
for row in rows:
    eid = row.get("example_id")

    if row.get("token_limit_reached") is True:
        status = "unscorable_token_limit"
    elif eid in overrides:
        status = overrides[eid]
    else:
        pass1 = row.get("manual_audit_status")
        pass2 = row.get("manual_audit_pass2_status")

        if pass1 == "provisional_correct":
            status = "provisional_correct"
        elif pass1 == "provisional_incorrect":
            status = "provisional_incorrect"
        elif pass2 == "provisional_correct":
            status = "provisional_correct"
        elif pass2 == "provisional_incorrect":
            status = "provisional_incorrect"
        else:
            status = "unresolved"

    row["consolidated_audit_status_v1"] = status
    counts[status] += 1

summary = {
    "version": "baseline_summary_v1",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "GSM8K test pilot",
    "pilot_size": len(rows),
    "counts": dict(sorted(counts.items())),
    "scorable_examples": len(rows) - counts["unscorable_token_limit"],
    "provisional_correct": counts["provisional_correct"],
    "provisional_incorrect": counts["provisional_incorrect"],
    "provisional_accuracy_percent": round(
        100 * counts["provisional_correct"] /
        max(1, counts["provisional_correct"] + counts["provisional_incorrect"]), 2
    ),
    "overrides": overrides,
    "caveat": (
        "Provisional accuracy only. Labels have not all been independently "
        "verified against full questions, complete model reasoning, and reference solutions."
    ),
}

with source.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

for row in rows:
    eid = row.get("example_id")
    if row.get("token_limit_reached") is True:
        status = "unscorable_token_limit"
    elif eid in overrides:
        status = overrides[eid]
    else:
        pass1 = row.get("manual_audit_status")
        pass2 = row.get("manual_audit_pass2_status")
        if pass1 == "provisional_correct" or pass2 == "provisional_correct":
            status = "provisional_correct"
        elif pass1 == "provisional_incorrect" or pass2 == "provisional_incorrect":
            status = "provisional_incorrect"
        else:
            status = "unresolved"
    row["consolidated_audit_status_v1"] = status

with (base / "baseline_qwen_0.5b_100_800tokens_baseline_audited_v1.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nCreated:", output)
print("Created:", base / "baseline_qwen_0.5b_100_800tokens_baseline_audited_v1.jsonl")
