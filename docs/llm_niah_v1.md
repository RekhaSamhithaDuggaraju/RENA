# RENA LLM and NIAH evaluation — initial implementation

## Scope

This module introduces a deterministic fictional fact set, NIAH case generation,
model-agnostic generation/scoring, and an optional Hugging Face causal-LM adapter.
It is an evaluation foundation, not yet a complete parameter-unlearning trainer.
The existing classical RENA track remains unchanged.

## Components

- `core/llm/knowledge.py`: immutable fictional `Fact` records and stable dataset hash.
- `core/llm/niah.py`: context/case generation and normalized exact-match scoring.
- `core/llm/hf_adapter.py`: optional local/Hugging Face generation adapter; imports
  torch and transformers lazily.
- `run_niah_benchmark.py`: executable benchmark runner that saves full per-case
  outputs and summary metrics as JSON.
- `requirements-llm.txt`: optional dependencies, isolated from core requirements.

## Run

Create and activate a dedicated virtual environment from the repository root:

```bash
python3 -m venv .venv-llm
source .venv-llm/bin/activate
python -m pip install -r requirements-llm.txt
python run_niah_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct --output results/llm/niah.json
```

The model ID above is an example. A compatible local model directory can be
passed instead. Downloading a model requires network access; hardware and memory
requirements depend on the selected model. On a Mac, `--device auto` attempts
MPS when available and otherwise uses CPU.

## Interpretation

- `forget_disclosure_rate`: proportion of forget-target cases whose normalized
  answer string appears in the model output. Lower is preferable.
- `retain_retrieval_rate`: proportion of retain-target cases whose normalized
  answer string appears in the output. Higher is preferable.
- Each case stores its target split, answer, prompt, output, context character
  length, insertion position and prompt style for traceability.

Context sizes count distractor records and are not token counts. Scoring uses
normalized exact match and can miss semantically equivalent or indirect
responses. NIAH measures retrieval from supplied context and cannot alone prove
that a model's parameters no longer retain information. Report it with direct
and paraphrased prompting, retained utility, recovery testing and an appropriate
retain-only retraining reference. Do not compare raw NIAH scores across models
without controlling tokenization, context limits, prompt format and decoding.

## Not yet implemented

This first increment does not fine-tune an LLM, apply GA/GD/NPO, integrate LLM
experiments into the web UI, or establish a validated semantic judge. These are
subsequent milestones; do not describe them as completed based on this module.

## Parametric-disclosure control and LLM training baselines

NIAH cases now support `context_modes=["present", "absent"]`. `present`
embeds the target fact in the supplied records and measures contextual retrieval.
`absent` omits the target fact from the records and probes whether the model
answers from learned parametric knowledge. These rates are reported separately;
context-present disclosure must not be interpreted as failed unlearning because
the answer was supplied in the prompt. Neither mode alone establishes physical
erasure or guarantees against indirect disclosure. Exact normalized match is a
conservative lexical metric, so manual review or a separately validated semantic
metric is needed for paraphrased disclosures.

## Controlled unlearning baselines

`core/llm/unlearning.py` adds optional causal-LM training baselines:

- `retain_finetuning`: minimize retain-fact answer cross-entropy.
- `gradient_ascent`: maximize forget-fact answer cross-entropy.
- `gradient_difference`: minimize retain cross-entropy minus forget cross-entropy.
- `npo`: lower forget-answer likelihood relative to a frozen copy of the initial
  model using a stable softplus objective.

Example (run from repository root, with optional dependencies installed):

```bash
python3 run_llm_unlearning.py --model /path/to/small-causal-lm \
  --method gradient_ascent --output results/llm/gradient_ascent
python3 run_niah_benchmark.py --model results/llm/gradient_ascent/checkpoint \
  --context-sizes 8 32 --context-modes present absent \
  --output results/llm/gradient_ascent/niah.json
```

Use the same starting checkpoint, facts, tokenization, training budget and seed
across methods. Evaluate the untouched base model and a separately trained
retain-only reference with the same NIAH configuration. The training runner
writes a checkpoint and `run_metadata.json`; it does not claim that training
completed successfully until the command finishes. NPO deep-copies the base
model and can require substantial memory. These simple full-parameter baselines
are intended for small controlled experiments; they are not a substitute for
reproducing each method's original paper-level recipe. Synthetic facts are for
pipeline validation and must be supplemented with a defensible public benchmark
before making broad claims.

The Hugging Face generation adapter does not silently truncate long NIAH
prompts. When model context metadata is available, it rejects prompts that
exceed the context limit after reserving generation tokens. Reduce the selected
context sizes rather than accepting a test whose inserted fact or question may
have been truncated.

The training runner first performs a controlled mixed-fact warmup for
fine-tuning, GA, GD and NPO, then applies the selected method. The
`retain_only_retraining` baseline starts from the supplied initialization and
trains only on retain facts. This is a controlled retain-only reference, not a
claim of exact equivalence to a production model's original training process.

## Multi-seed analysis

Run `run_llm_suite.py` separately for each predeclared seed, with the same base model, knowledge generation protocol, methods, hyperparameters, context sizes and generation settings. Store each run in its own directory. Then aggregate the resulting `suite_summary.json` files with `run_llm_aggregate.py`. The aggregator reports mean, sample standard deviation and a percentile bootstrap interval across seeds, plus paired per-seed differences against `retain_only_retraining` when that method is present. Do not treat prompt cases within a seed as independent samples for seed-level uncertainty. With very few seeds, intervals are descriptive and should not be used to imply broad generalization.

Example:

```bash
python run_llm_suite.py --model <MODEL_ID_OR_PATH> --seed 42 --output results/llm/seed_42
python run_llm_suite.py --model <MODEL_ID_OR_PATH> --seed 43 --output results/llm/seed_43
python run_llm_suite.py --model <MODEL_ID_OR_PATH> --seed 44 --output results/llm/seed_44
python run_llm_aggregate.py results/llm/seed_42/suite_summary.json results/llm/seed_43/suite_summary.json results/llm/seed_44/suite_summary.json --output results/llm/aggregate.json
```

Each seed run reloads each method from the supplied base checkpoint. The current synthetic-fact suite is a controlled engineering benchmark, not a replacement for established LLM unlearning benchmarks. Exact-match retrieval metrics are lexical proxies and do not establish semantic non-disclosure or physical erasure. Report actual model ID/revision, hardware, library versions, context settings, seeds, failures and compute costs with any results.
