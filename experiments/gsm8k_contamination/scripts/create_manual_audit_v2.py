import json
from pathlib import Path
from collections import Counter

base = Path("experiments/gsm8k_contamination/data/pilot")
src = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_v1.jsonl"
out = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_v2.jsonl"
summary_path = base / "baseline_qwen_0.5b_100_800tokens_manual_audit_summary_v2.json"

# Provisional decisions based on the supplied reference and response endings.
# Question/reference ambiguities are deliberately left unresolved.
correct = """
0001 0004 0005 0008 0010 0011 0015 0023 0026 0033
0037 0038 0041 0043 0044 0046 0048 0051 0055 0059
0065 0066 0072 0073 0082 0084 0085 0088 0090 0092
0093 0095 0096
""".split()

incorrect = """
0000 0006 0007 0016 0017 0021 0032 0035 0036 0042
0045 0049 0050 0052 0053 0058 0063 0067 0069 0071
0075 0076 0077 0086 0087 0089 0091 0097 0098
""".split()

needs_check = {
    "0022": "The response gives 14 but the supplied reference is 24. Inspect the full question and answer key before finalizing.",
    "0094": "The response gives 15 but the supplied reference is 50. Inspect the full question and answer key before finalizing.",
}

token_limited = {
    "0099": "Generation hit the token limit and the response is incomplete; do not score this row."
}

if not src.exists():
    raise SystemExit(f"Missing input: {src}")

with src.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if len(rows) != 100:
    raise SystemExit(f"Expected 100 rows, found {len(rows)}. No output written.")

decisions = {}
for eid in correct:
    decisions[eid] = (
        "provisional_correct",
        "Final response matches the supplied reference; numeric units were considered where applicable."
    )
for eid in incorrect:
    decisions[eid] = (
        "provisional_incorrect",
        "Final response differs from the supplied reference."
    )
for eid, reason in needs_check.items():
    decisions[eid] = ("needs_question_reference_check", reason)
for eid, reason in token_limited.items():
    decisions[eid] = ("unscorable_token_limit", reason)

if len(decisions) != len(set(decisions)):
    raise SystemExit("Duplicate decision IDs found. No output written.")

found = {
    str(r.get("example_id", "")).replace("gsm8k_test_", "")
    for r in rows
}
missing = sorted(set(decisions) - found)
if missing:
    raise SystemExit(f"IDs missing from input: {missing}. No output written.")

result = []
for original in rows:
    r = dict(original)
    eid = str(r.get("example_id", "")).replace("gsm8k_test_", "")
    if eid in decisions:
        status, reason = decisions[eid]
        r["manual_audit_pass2_status"] = status
        r["manual_audit_pass2_reason"] = reason
    else:
        r["manual_audit_pass2_status"] = "not_reviewed_in_pass2"
        r["manual_audit_pass2_reason"] = None
    result.append(r)

with out.open("w", encoding="utf-8") as f:
    for r in result:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

counts = Counter(r["manual_audit_pass2_status"] for r in result)
summary = {
    "audit_version": "v2",
    "input_file": str(src),
    "output_file": str(out),
    "total_rows": len(result),
    "pass2_provisional_correct": counts["provisional_correct"],
    "pass2_provisional_incorrect": counts["provisional_incorrect"],
    "needs_question_reference_check": counts["needs_question_reference_check"],
    "unscorable_token_limit": counts["unscorable_token_limit"],
    "not_reviewed_in_pass2": counts["not_reviewed_in_pass2"],
    "note": (
        "Pass 2 labels are provisional and based on the supplied response endings "
        "and reference answers. They are not a substitute for checking full questions "
        "or independently verifying the answer key. Pass 1 fields are preserved."
    )
}

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(json.dumps(summary, indent=2))
print("\nCreated:", out)
print("Created:", summary_path)
print("V1 and all original scorer files remain unchanged.")
