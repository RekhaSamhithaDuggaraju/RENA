import json
import sys
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("experiments/gsm8k_contamination")
INPUT = ROOT / "data/pilot/pilot_clean.jsonl"
OUTPUT = ROOT / "data/pilot/baseline_qwen_0.5b_100_800tokens.jsonl"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 800
SEED = 20261025

if OUTPUT.exists():
    raise SystemExit(f"Refusing to overwrite existing output: {OUTPUT}")
if not INPUT.exists():
    raise SystemExit(f"Missing input dataset: {INPUT}")

rows = [
    json.loads(line)
    for line in INPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(rows) != 100:
    raise SystemExit(f"Expected 100 pilot rows; found {len(rows)}")

if len({row["example_id"] for row in rows}) != len(rows):
    raise SystemExit("Duplicate example IDs in pilot dataset")

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device == "mps" else torch.float32

print(f"Model: {MODEL_ID}")
print(f"Device: {device}")
print(f"Dtype: {dtype}")
print(f"Examples: {len(rows)}")
print("Decoding: greedy")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print("Prompt: one user message containing the GSM8K question; assistant generation follows the model chat template.")
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    local_files_only=True,
)
model.to(device)
model.eval()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("x", encoding="utf-8") as out:
    for index, row in enumerate(rows, start=1):
        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following grade-school math problem. "
                    "Show your reasoning briefly and finish with a line "
                    "in the exact format FINAL: <number>.\n\n"
                    f"{row['question']}"
                ),
            }
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

        prompt_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0, prompt_length:]
        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        record = {
            "example_id": row["example_id"],
            "question": row["question"],
            "reference_final_answer": row["answer"].split("####")[-1].strip(),
            "model_response": response,
            "model": MODEL_ID,
            "device": device,
            "dtype": str(dtype),
            "generation": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "generated_tokens": int(new_tokens.shape[0]),
            "token_limit_reached": int(new_tokens.shape[0]) >= MAX_NEW_TOKENS,
            "prompt_template": "Qwen chat template; one user message; explicit FINAL:number instruction",
            "prompt_sha256": hashlib.sha256(
                messages[0]["content"].encode("utf-8")
            ).hexdigest(),
            "seed": SEED,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()

        print(
            f"[{index:03d}/{len(rows)}] {row['example_id']} "
            f"generated_tokens={record['generated_tokens']} "
            f"token_limit_reached={record['token_limit_reached']}",
            flush=True,
        )

print(f"\nSaved: {OUTPUT}")
print("BASELINE_RUN=PASS")
