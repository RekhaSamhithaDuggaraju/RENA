import json
from pathlib import Path

base = Path("experiments/gsm8k_contamination/data/pilot")
source = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_corrected_v1.jsonl"
output = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_final_v1.jsonl"
summary_path = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_final_summary_v1.json"

manual_decisions = {
    "gsm8k_test_0012": {
        "status": "incorrect",
        "reference_answer": "9",
        "reason": "42 regular chairs needed and 15 high chairs needed; existing chairs are 40 regular and 8 high, so 2 + 7 = 9 additional chairs."
    },
    "gsm8k_test_0022": {
        "status": "incorrect",
        "reference_answer": "24",
        "reason": "John was 8 when Digimon came out; Jim was 4 then, so Jim is 24 now."
    },
    "gsm8k_test_0094": {
        "status": "incorrect",
        "reference_answer": "50",
        "reason": "Discounted total cost is $90; total resale proceeds are $40; net amount spent is $90 - $40 = $50."
    }
}

rows = []
with source.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        decision = manual_decisions.get(row.get("example_id"))
        if decision:
            row["verified_manual_decision_v1"] = decision
        rows.append(row)

with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "version": "final_v1",
    "total_rows": len(rows),
    "manual_decisions_added": len(manual_decisions),
    "decisions": manual_decisions,
    "note": (
        "This file consolidates the audit and adds verified decisions for the three "
        "previously disputed cases. It does not automatically certify every other row."
    )
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nCreated: {output}")
print(f"Created: {summary_path}")
