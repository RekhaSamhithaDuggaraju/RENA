import json
from pathlib import Path

base = Path("experiments/gsm8k_contamination/data/pilot")
source = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_v2.jsonl"
output = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_corrected_v1.jsonl"
summary_path = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_corrected_summary_v1.json"

corrections = {
    "gsm8k_test_0022": {
        "status": "confirmed_incorrect",
        "reason": (
            "The model ignored the 20-year interval. John was 8 when Digimon "
            "came out; Jim was 4 then, so Jim is 24 now."
        ),
        "verified_reference_answer": "24",
    },
    "gsm8k_test_0094": {
        "status": "reference_check_required",
        "reason": (
            "The model's reasoning is inconsistent. Under the literal wording, "
            "the discounted cost is $90 and selling five CDs for $40 each yields "
            "a $110 net gain. The supplied reference answer $50 does not follow "
            "from that interpretation; verify the original GSM8K question and solution."
        ),
        "verified_reference_answer": None,
    },
}

rows = []
with source.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        correction = corrections.get(row.get("example_id"))
        if correction:
            row["audit_correction_v1"] = correction
        rows.append(row)

with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "version": "corrected_v1",
    "total_rows": len(rows),
    "corrections_recorded": len(corrections),
    "corrections": corrections,
    "note": (
        "Original V1/V2 audit files are unchanged. Case 0094 remains excluded "
        "from confirmed accuracy until the reference solution is checked."
    ),
}

summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nCreated: {output}")
print(f"Created: {summary_path}")
