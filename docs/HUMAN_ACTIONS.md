# Human Action Items & External Dependencies

This document enumerates the required human and out-of-band operational actions that cannot be simulated or automated by the AI assistant under Rule 3 ("No simulation presented as evidence, no invented labels") and Rule 8 ("Humans cannot be simulated").

---

## Required Human Actions

### Action 1: Independent Dual-Rater Annotation of Real Auditing Logs
- **Task**: Independently label `data/annotation/annotator_A.csv` and `data/annotation/annotator_B.csv` without consulting each other or looking at the model's self-verdicts.
- **Scope**: Approximately 600 real response items sampled and stratified across disclosure classes (clear disclosure, partial/indirect, near-miss, refusal, refusal-with-token-mention, hallucinated, unrelated).
- **Time Required**: ~2–3 hours per annotator.
- **Next Step**: Once both CSV files have `human_label` filled, execute:
  ```bash
  python3 scripts/analyze_annotation.py --rater-a data/annotation/annotator_A.csv --rater-b data/annotation/annotator_B.csv
  ```
  This computes Cohen's Kappa, inter-rater agreement, precision/recall/F1 with bootstrap confidence intervals, and flags disputed cases for consensus adjudication.

### Action 2: Multi-Provider API Configuration (Optional)
- **Task**: Supply an API key for a second commercial provider (e.g. Anthropic, OpenAI) in `.env` if desired:
  ```bash
  ANTHROPIC_API_KEY=your_key_here
  # or
  OPENAI_API_KEY=your_key_here
  ```
- **Scope**: The hosted model audit currently runs on Google Gemini (`gemini-2.0-flash-lite`). If no second key is provided, the study honestly reports single-provider black-box scope as an explicit limitation in Section 5 of the manuscript.

### Action 3: Review and Formal Sign-Off on Preregistration
- **Task**: Inspect `docs/PREREGISTRATION.md` and approve the registered hypotheses, sample sizes ($n \ge 100$ per condition across $\ge 3$ seeds), stopping criteria, and the conditional pivot rule ($H_{\text{main}}$) before running confirmatory test seeds.
- **Falsification Thresholds**: Verify that the pre-registered false-pass rate threshold ($\alpha = 0.05$) and equivalence bounds ($\delta = 0.05$) match your experimental standards.

### Action 4: Google Colab Run for Second Architecture (If Locally Constrained)
- **Task**: If running a larger 1.5B–3B parameter model (e.g. `Qwen2.5-1.5B-Instruct` or `Llama-3.2-1B`) exceeds Apple Silicon unified memory during full gradient unlearning or batching, execute:
  `notebooks/testbed_colab.ipynb` on Google Colab (T4 / A100 GPU).
- **Scope**: Download the resulting `results/testbed/second_model_audit.json` into `results/testbed/`. If not executed, the single-model limitation remains declared in the paper.

### Action 5: Final Bibliography & Related Work Verification
- **Task**: Verify the 8 core citations in `paper/references.bib` against the published papers:
  1. Eldan & Russinovich (2023) — *Who's Harry Potter?*
  2. Maini et al. (2024) — *TOFU Benchmark*
  3. Shi et al. (2024) — *MUSE Benchmark*
  4. Li et al. (2024) — *WMDP Benchmark*
  5. Carlini et al. (2021) — *Extracting Training Data from Large Language Models*
  6. Zheng et al. (2023) — *Judging LLM-as-a-Judge*
  7. Bowman et al. (2023) — *Measuring Faithfulness in Chain-of-Thought*
  8. Dwork et al. (2006) — *Differential Privacy*
- **Check**: Ensure every factual attribution in `paper/main.tex` accurately reflects the cited authors' claims.
