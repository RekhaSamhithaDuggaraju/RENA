import json
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("experiments/gsm8k_contamination")
DATA = ROOT / "data/pilot"
SCRIPT_VERSION = "perturbation_inference_v1"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 800
SEED = 20261025

CONDITIONS = ["wording", "formatting"]

baseline_path = DATA / "baseline_qwen_0.5b_100_800tokens.jsonl"
if not baseline_path.exists():
    raise SystemExit(f"Missing baseline output: {baseline_path}")

baseline = {
    r["example_id"]: r
    for r in (
        json.loads(line)
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
}

inputs = {}
outputs = {}

for condition in CONDITIONS:
    input_path = DATA / "perturbations" / f"{condition}.jsonl"
    output_path = DATA / f"{condition}_changed_qwen_0.5b_v1.jsonl"

    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    changed = [
        row for row in rows
        if row["question"] != row["original_question"]
    ]

    if any(row["example_id"] not in baseline for row in changed):
        raise SystemExit(f"Some {condition} IDs are missing from baseline")

    inputs[condition] = changed
    outputs[condition] = output_path

    print(f"{condition}: {len(changed)} changed questions")

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32

print(f"Model: {MODEL_ID}")
print(f"Device: {device}; dtype: {dtype}")
print("Loading local model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    local_files_only=True,
)
model.to(device)
model.eval()

for condition in CONDITIONS:
    output_path = outputs[condition]
    rows = inputs[condition]

    print(f"\n=== CONDITION: {condition} ===")

    with output_path.open("x", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            prompt = (
                "Solve the following grade-school math problem. "
                "Show your reasoning briefly and finish with a line "
                "in the exact format FINAL: <number>.\n\n"
                f"{row['question']}"
            )

            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            prompt_length = model_inputs["input_ids"].shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0, prompt_length:]
            response = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            record = {
                "example_id": row["example_id"],
                "condition": condition,
                "question": row["question"],
                "original_question": row["original_question"],
                "reference_final_answer": row["reference_answer"].split("####")[-1].strip(),
                "model_response": response,
                "model": MODEL_ID,
                "device": device,
                "dtype": str(dtype),
                "generation": "greedy",
                "max_new_tokens": MAX_NEW_TOKENS,
                "generated_tokens": int(new_tokens.shape[0]),
                "token_limit_reached": int(new_tokens.shape[0]) >= MAX_NEW_TOKENS,
                "prompt_template": "Qwen chat template; one user message; explicit FINAL:number instruction",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "seed": SEED,
                "script_version": SCRIPT_VERSION,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(
                f"[{index}/{len(rows)}] {row['example_id']} "
                f"tokens={record['generated_tokens']} "
                f"limit={record['token_limit_reached']}",
                flush=True,
            )

    print(f"Saved: {output_path}")

print("\nPERTURBATION_INFERENCE=PASS")
