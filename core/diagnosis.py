"""
RENA core — Failure Diagnosis Engine.

Rule-based (not another ML model) so its reasoning is fully inspectable.
Every statement it produces is tied to a specific number computed upstream.
"""
from __future__ import annotations

from core.verification import ModuleResult
from core.consensus import ConsensusResult


def diagnose(consensus: ConsensusResult, module_results: list[ModuleResult],
             accuracy_retain: float, accuracy_test: float) -> dict:
    findings = []
    recommendations = []

    if consensus.verdict == "Inconclusive":
        findings.append(
            f"Verification modules disagree (agreement={consensus.agreement:.2f}, "
            f"below the {0.5} threshold), so no confident verdict can be issued even "
            f"though the average forget score is {consensus.consensus_score:.2f}."
        )
        # identify which module(s) are the outlier
        scores = [(m.module_name, m.forget_score) for m in module_results]
        scores_sorted = sorted(scores, key=lambda x: x[1])
        spread = scores_sorted[-1][1] - scores_sorted[0][1]
        if spread > 0.4:
            findings.append(
                f"Largest disagreement: '{scores_sorted[0][0]}' scored "
                f"{scores_sorted[0][1]:.2f} while '{scores_sorted[-1][0]}' scored "
                f"{scores_sorted[-1][1]:.2f} — these tests disagree on whether "
                f"forgetting occurred."
            )
        recommendations.append(
            "Increase forget-set size if possible, or run additional verification "
            "rounds with different random seeds to check whether disagreement is "
            "due to sampling noise rather than a genuine leak."
        )

    if consensus.verdict == "Not Forgotten":
        mia = next((m for m in module_results if m.module_name == "Membership Inference Attack"), None)
        if mia and mia.raw_statistic > 0.65:
            findings.append(
                f"Membership Inference Attack achieved AUC={mia.raw_statistic:.2f} "
                f"(well above the 0.5 chance baseline), meaning forget-set samples "
                f"remain distinguishable from never-seen data — a direct privacy leak."
            )
            recommendations.append(
                "The unlearning algorithm did not sufficiently perturb the model's "
                "decision boundary around the forget set. Consider gradient ascent "
                "with more epochs, or full retraining if compute allows."
            )
        loss_mod = next((m for m in module_results if m.module_name == "Loss Distribution KS-Test"), None)
        if loss_mod and loss_mod.p_value is not None and loss_mod.p_value < 0.05:
            findings.append(
                f"Loss distribution KS-test p-value={loss_mod.p_value:.4f} < 0.05: "
                f"the forget set's loss distribution is statistically distinguishable "
                f"from the test set's, indicating residual memorization."
            )

    if accuracy_retain < accuracy_test - 0.05:
        findings.append(
            f"Retain-set accuracy ({accuracy_retain:.2f}) dropped notably below "
            f"test-set accuracy ({accuracy_test:.2f}), suggesting the unlearning "
            f"procedure damaged general model utility, not just forget-set memorization."
        )
        recommendations.append(
            "Reduce the aggressiveness of the unlearning step (fewer ascent epochs, "
            "lower learning rate) or add a longer recovery fine-tune on the retain set."
        )

    for dm in consensus.discounted_modules:
        if dm["reliability"] < 0.6:
            findings.append(f"'{dm['module']}' reliability was discounted to "
                             f"{dm['reliability']:.2f}: {dm['reason']}.")

    if not findings:
        findings.append(
            f"No failure indicators detected. Consensus verdict '{consensus.verdict}' "
            f"is supported by agreement={consensus.agreement:.2f} across "
            f"{len(module_results)} complementary verification witnesses."
        )

    return {
        "verdict": consensus.verdict,
        "findings": findings,
        "recommendations": recommendations or ["No corrective action indicated."],
    }
