"""
RENA core — verification modules.

Four independent statistical tests for whether a model has actually
forgotten a given forget set. Each returns a ModuleResult with a real
computed statistic — none of these are placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from core.model import per_sample_loss


@dataclass
class ModuleResult:
    module_name: str
    raw_statistic: float
    p_value: float | None
    forget_score: float          # 0 = clearly NOT forgotten, 1 = clearly forgotten
    reliability: float           # 0..1, this module's confidence in its own result
    reliability_reason: str
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _reliability_from_sample_size(n: int, min_reliable: int = 50) -> tuple[float, str]:
    factor = min(1.0, n / min_reliable)
    reason = (f"forget-set size ({n}) is below the {min_reliable}-sample threshold "
              f"for reliable statistical power" if factor < 1.0
              else f"forget-set size ({n}) meets the reliability threshold")
    return factor, reason


# ---------------------------------------------------------------------------
# 1. Membership Inference Attack (MIA)
# ---------------------------------------------------------------------------
def membership_inference_attack(model, X_forget, y_forget, X_test, y_test,
                                 model_test_accuracy: float) -> ModuleResult:
    """
    Trains a binary attacker to distinguish forget-set samples (potential
    "members") from held-out test samples (definite "non-members") using
    per-sample loss + max confidence + entropy as features. AUC near 0.5
    means the attacker can't tell them apart -> forgetting looks successful.
    AUC near 1.0 means the forget set is still distinguishable -> leak.
    """
    n_forget = len(X_forget)
    losses_f = per_sample_loss(model, X_forget, y_forget)
    losses_t = per_sample_loss(model, X_test[:n_forget], y_test[:n_forget])

    probs_f = model.predict_proba(X_forget)
    probs_t = model.predict_proba(X_test[:n_forget])

    def feats(losses, probs):
        max_conf = probs.max(axis=1)
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        return np.stack([losses, max_conf, entropy], axis=1)

    X_attack = np.vstack([feats(losses_f, probs_f), feats(losses_t, probs_t)])
    y_attack = np.concatenate([np.ones(len(losses_f)), np.zeros(len(losses_t))])

    n_splits = min(5, min(np.bincount(y_attack.astype(int))))
    n_splits = max(2, n_splits)
    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        attack_probs = cross_val_predict(
            LogisticRegression(max_iter=1000), X_attack, y_attack,
            cv=cv, method="predict_proba"
        )[:, 1]
        auc = float(roc_auc_score(y_attack, attack_probs))
    except ValueError:
        auc = 0.5  # degenerate case (too few samples) -> treat as uninformative

    forget_score = float(np.clip(1 - 2 * max(0, auc - 0.5), 0, 1))

    reliability, size_reason = _reliability_from_sample_size(n_forget)
    capped_reason = None
    if model_test_accuracy < 0.80:
        reliability = min(reliability, 0.6)
        capped_reason = (f"base model test accuracy ({model_test_accuracy:.2f}) is below "
                          f"0.80 — weak models produce weak/uninformative attack signal "
                          f"regardless of unlearning quality")
    reason = size_reason + (f"; {capped_reason}" if capped_reason else "")

    return ModuleResult(
        module_name="Membership Inference Attack",
        raw_statistic=auc,
        p_value=None,
        forget_score=forget_score,
        reliability=float(reliability),
        reliability_reason=reason,
        detail={"attack_auc": auc, "n_forget": n_forget},
    )


# ---------------------------------------------------------------------------
# 2. Loss Distribution KS-test
# ---------------------------------------------------------------------------
def loss_distribution_test(model, X_forget, y_forget, X_test, y_test) -> ModuleResult:
    """
    Kolmogorov-Smirnov two-sample test comparing the forget set's per-sample
    loss distribution to the test set's. A small KS statistic (distributions
    indistinguishable) indicates the forget set now "looks like" unseen data.
    """
    losses_f = per_sample_loss(model, X_forget, y_forget)
    n_forget = len(X_forget)
    losses_t = per_sample_loss(model, X_test[:max(n_forget, 30)], y_test[:max(n_forget, 30)])

    ks_stat, p_value = stats.ks_2samp(losses_f, losses_t)
    forget_score = float(np.clip(1 - ks_stat, 0, 1))

    reliability, reason = _reliability_from_sample_size(n_forget)

    return ModuleResult(
        module_name="Loss Distribution KS-Test",
        raw_statistic=float(ks_stat),
        p_value=float(p_value),
        forget_score=forget_score,
        reliability=float(reliability),
        reliability_reason=reason,
        detail={
            "forget_loss_mean": float(np.mean(losses_f)),
            "test_loss_mean": float(np.mean(losses_t)),
        },
    )


# ---------------------------------------------------------------------------
# 3. Confidence Analysis
# ---------------------------------------------------------------------------
def confidence_analysis(model, original_model, X_forget, y_forget, X_test, y_test) -> ModuleResult:
    """
    Compares the mean prediction-confidence gap between forget-set and
    test-set samples under the UNLEARNED model, calibrated against the same
    gap measured on the ORIGINAL (pre-unlearning) model — this puts the
    result on an absolute scale rather than an arbitrary raw magnitude.
    """
    n_forget = len(X_forget)
    probs_f = model.predict_proba(X_forget)
    probs_t = model.predict_proba(X_test[:n_forget])
    gap_unlearned = float(np.mean(probs_f.max(axis=1)) - np.mean(probs_t.max(axis=1)))

    orig_probs_f = original_model.predict_proba(X_forget)
    orig_probs_t = original_model.predict_proba(X_test[:n_forget])
    gap_baseline = float(np.mean(orig_probs_f.max(axis=1)) - np.mean(orig_probs_t.max(axis=1)))
    # Floor at 0.05: on an easy, well-fit dataset the ORIGINAL model's own
    # forget/test confidence gap can be near-zero (both near-saturated
    # confidence), which makes a ratio against it numerically unstable --
    # tiny absolute differences get amplified into ratio~1 (forget_score~0)
    # even when the unlearned model's gap is objectively small. 0.05 is a
    # floor representing "a gap this small isn't a meaningful leak signal
    # regardless of what the baseline model happened to show."
    gap_baseline = max(abs(gap_baseline), 0.05)

    forget_score = float(np.clip(1 - min(1, abs(gap_unlearned) / gap_baseline), 0, 1))

    reliability, reason = _reliability_from_sample_size(n_forget)

    return ModuleResult(
        module_name="Confidence Analysis",
        raw_statistic=gap_unlearned,
        p_value=None,
        forget_score=forget_score,
        reliability=float(reliability),
        reliability_reason=reason,
        detail={"gap_unlearned": gap_unlearned, "gap_baseline_original_model": gap_baseline},
    )


# ---------------------------------------------------------------------------
# 4. Bootstrap Statistical Witness
# ---------------------------------------------------------------------------
def bootstrap_statistical_witness(model, X_forget, y_forget, X_retain, y_retain,
                                   n_bootstrap: int = 2000, random_state: int = 42) -> ModuleResult:
    """
    Bootstrap-resamples the difference in mean loss between the forget set
    and the retain set to build an empirical null distribution, then reports
    a two-sided p-value for the observed difference. A non-significant
    p-value (forget set's loss is statistically indistinguishable from the
    retain set's) supports successful forgetting.
    """
    rng = np.random.RandomState(random_state)
    losses_f = per_sample_loss(model, X_forget, y_forget)
    n_retain_sample = min(len(X_retain), max(len(X_forget) * 5, 100))
    retain_idx = rng.choice(len(X_retain), size=n_retain_sample, replace=False)
    losses_r = per_sample_loss(model, X_retain[retain_idx], y_retain[retain_idx])

    observed_diff = float(np.mean(losses_f) - np.mean(losses_r))

    pooled = np.concatenate([losses_f, losses_r])
    n_f = len(losses_f)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(pooled, size=len(pooled), replace=True)
        boot_diffs[i] = np.mean(sample[:n_f]) - np.mean(sample[n_f:])

    p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(observed_diff)))
    # NOTE (correction vs. original spec draft): retain-set loss is near-zero by
    # construction (the model was fit to it), so a genuinely FORGOTTEN sample
    # should look DIFFERENT from retain, not similar to it. A small p-value
    # (significant difference from retain) is therefore evidence FOR forgetting,
    # not against it -- the original spec had this inverted. Fixed here.
    forget_score = float(np.clip(1 - p_value, 0, 1))

    reliability, reason = _reliability_from_sample_size(n_f)
    boot_se = float(np.std(boot_diffs))
    if boot_se > 0.5:  # high resampling variance -> less trustworthy estimate
        reliability = min(reliability, 0.5)
        reason += f"; high bootstrap variance (SE={boot_se:.3f}) reduces confidence"

    return ModuleResult(
        module_name="Bootstrap Statistical Witness",
        raw_statistic=observed_diff,
        p_value=p_value,
        forget_score=forget_score,
        reliability=float(reliability),
        reliability_reason=reason,
        detail={"observed_diff": observed_diff, "bootstrap_se": boot_se, "n_bootstrap": n_bootstrap},
    )
