# ECHO experiment report

## Real experimental results
ECHO seed-42 consensus: 0.7066; agreement: 0.7978; forget/retain/test accuracy: 1.0000/1.0000/0.9578.

## Method
ECHO is a closed-loop mechanism: four fast proxies are evaluated for temporary NumPy-MLP candidate updates; non-dominated candidates are selected by Pareto dominance and deterministic utility/recovery/evidence/norm tie-breaks. Full RENA verification runs after ECHO completion. This is evidence-guided verification feedback, not a guarantee of unlearning.

## Real Gemini results / fallback / API errors
{
  "overall_status": "SUPPORTED",
  "supporting_evidence_ids": [
    "E1001",
    "E1002",
    "E1003",
    "E1004"
  ],
  "contradictory_evidence_ids": [],
  "missing_evidence": [
    "adaptive recovery attack",
    "independent external replication"
  ],
  "limitations": [
    "OFFLINE FALLBACK: GEMINI_API_KEY was absent; no Gemini API call was made.",
    "Evidence is limited to tested digits data and perturbations."
  ],
  "claim_strength": "MODERATE",
  "claim": "The structured verification evidence supports forgetting only under the tested conditions.",
  "reasoning_summary": "Rule-based synthesis of supplied evidence IDs; no metrics were generated or altered.",
  "analysis_mode": "FALLBACK",
  "provider": "Offline fallback",
  "model": null
}

## Pending experiments and limitations
The adaptive recovery test is a lightweight confidence/loss-guided perturbation sweep, not a state-of-the-art adaptive attack. Gemini is not a real API experiment when its mode is FALLBACK. Repeats use three train/split seeds and report sample standard deviations.
