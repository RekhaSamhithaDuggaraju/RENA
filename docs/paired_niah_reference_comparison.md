# Paired NIAH comparison against retain-only retraining

`compare_niah_runs.py` compares candidate and retain-only reference NIAH result files using shared `case_id` values. It rejects duplicate IDs, invalid result fields, and mismatched case metadata. Unmatched cases are reported but excluded from paired estimates.

For each evaluation slice it reports candidate and reference retrieval rates and their paired difference (candidate minus reference). The 95% interval uses a bootstrap over `fact_id` clusters rather than treating multiple prompt variants for one fact as independent observations.

Example:

```bash
python3 compare_niah_runs.py \
  --candidate results/llm/gradient_ascent/niah_full.json \
  --reference results/llm/retain_only_retraining/niah_full.json \
  --candidate-name gradient_ascent \
  --output results/llm/gradient_ascent/paired_reference.json
```

Run both files with the same model, seed, fact set, context sizes, positions, prompt styles, context modes, and decoding configuration. Prefer full matched NIAH grids. The result is a behavioral comparison, not evidence of physical parameter deletion. The bootstrap interval is descriptive and can be unstable when there are few unique facts. Use multiple independent seeds for broader uncertainty analysis.
