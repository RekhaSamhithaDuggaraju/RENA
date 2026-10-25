# RENA Architectural Changelog

## [Version 2.3.0] — Research-Grade Transformation & Two-Track Architecture

### Core Scientific Contributions
- **Two-Track Audit Architecture**: Introduced distinct protocols for **Track A (Model-Level Weight Unlearning)** vs. **Track B (Application Memory & Retrieval Deletion)** to eliminate scientific conflation of parameter modifications and store-level deletions.
- **3-Group Experimental Architecture**: Implemented Group A (Target), Group B (Never-Exposed Reference Control), and Group C (Exposed Without Forgetting Counterfactual Control).
- **Causal Attribution Delta ($\Delta_{\text{causal}}$)**: Added formal statistical estimation of causal reduction in target recovery ($\Delta_{\text{causal}} = \text{Recovery}_{\text{Group C}} - \text{Recovery}_{\text{Group A}}$) with standard error and paired confidence intervals.
- **Wilson Score 95% Confidence Intervals**: Replaced simplistic percentage reporting with sample-size-calibrated Wilson score intervals ($k / N$) on all primary rates.
- **5 Experiment Validity Gates**: Added pre-execution and reporting validity checks (Dataset Boundaries, Baseline Retrievability, Operation Documentation, Matched Controls, Probe Diversity) with `PASS`, `WARNING`, and `BLOCKED` states.

### Backend Improvements
- **Exponential Backoff Retry**: Added resilient HTTP client retry logic for transient API rate limits (429) across Gemini, OpenAI, Anthropic, and Ollama.
- **Dataset Filtering Studio**: Added `POST /{aid}/dataset/filter` allowing researchers to slice datasets by index range and limit before running full batches.
- **Refusal & Hedging Classifier**: Regex classifier distinguishing `clean_refusal`, `refusal_with_token_mention`, `unhedged_disclosure`, and `no_disclosure`.
- **Publication LaTeX Exporter**: Added `GET /{aid}/export/latex` and embedded LaTeX table generator producing Overleaf-ready `tabular` code for both Track A and Track B.

### Frontend Polish
- **Track Selection on Workspace Creation**: Clear choice between Model Unlearning and Application Memory Deletion with track-specific parameter configurations.
- **Validity Gates Visualizer**: Interactive indicator showing status across all 5 scientific gates.
- **3-Group Comparison Metric Cards**: Highlighting Group A, B, and C with Wilson 95% CIs and sample denominators.
- **Direct Authenticated Downloads**: One-click download for CSV, JSON (SHA-256), and LaTeX `.tex` files with zero 401 token authentication errors.
