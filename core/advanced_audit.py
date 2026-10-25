"""RENA-X reference-distribution research audit.

RENA-X is a research audit, not a formal certificate.  A single retraining
run is not treated as the definition of a good model: when a reference
ensemble is supplied, every axis is calibrated against the empirical
variability of independently retrained retain-only models.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from core.model import evaluate, per_sample_loss, clone_model
from core.probes import discontinuity_profile, discontinuity_score


@dataclass
class AdvancedAuditResult:
    behavioral_equivalence: float
    membership_indistinguishability: float
    representation_neutrality: float
    retain_consistency: float
    irreversibility: float
    efficiency: float
    worst_case_score: float
    correctness_worst_case_score: float
    parameter_distance: float
    jsd_to_retraining: float
    representation_auc: float
    representation_probe_auc: float
    forget_progress: float
    retain_utility_ratio: float
    intervention_leakage: float
    utility_drop: float
    runtime_seconds: float
    gold_runtime_seconds: float
    efficiency_included_in_verdict: bool = False
    scoring_basis: str = "reference-relative; optionally calibrated to an independent retrain distribution"
    caveat: str = "Research audit only; not a formal proof of physical erasure, privacy, or differential privacy."
    calibration: dict | None = None
    discontinuity_consistency: float = 1.0
    discontinuity_profile: dict | None = None

    def to_dict(self):
        return asdict(self)


def _safe_auc(features_a, features_b):
    n=min(len(features_a), len(features_b))
    if n < 4: return 0.5
    X=np.vstack([features_a[:n], features_b[:n]])
    y=np.r_[np.ones(n), np.zeros(n)]
    n_splits=min(5, n)
    try:
        cv=StratifiedKFold(n_splits=max(2,n_splits),shuffle=True,random_state=2718)
        probe=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs'))
        pred=cross_val_predict(probe,X,y,cv=cv,method='predict_proba')[:,1]
        return float(roc_auc_score(y,pred))
    except ValueError:
        return 0.5


def _dist(a,b):
    return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))


def _proba(model,X): return model.predict_proba(X)


def _representation(model,X):
    if hasattr(model,"hidden_representation"):
        return model.hidden_representation(X)
    if hasattr(model,"shard_models") and model.shard_models:
        return np.mean([m.hidden_representation(X) for m in model.shard_models],axis=0)
    return _proba(model,X)


def _parameter_vector(model):
    if hasattr(model,"parameter_vector"): return model.parameter_vector()
    if hasattr(model,"shard_models"):
        return np.concatenate([m.parameter_vector() for m in model.shard_models])
    return None


def _linear_cka(X, Y):
    """Linear CKA; invariant to orthogonal/permutation changes of features."""
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return 0.0
    X=X-X.mean(axis=0,keepdims=True); Y=Y-Y.mean(axis=0,keepdims=True)
    K=X@X.T; L=Y@Y.T
    denom=np.linalg.norm(K,'fro')*np.linalg.norm(L,'fro')
    return float(np.clip(np.sum(K*L)/(denom+1e-12),0,1))


def _membership_auc(model, split):
    n=len(split.X_forget)
    lf=per_sample_loss(model,split.X_forget,split.y_forget)
    lt=per_sample_loss(model,split.X_test[:n],split.y_test[:n])
    pf=model.predict_proba(split.X_forget); pt=model.predict_proba(split.X_test[:n])
    def feats(loss,p):
        conf=p.max(axis=1); ent=-np.sum(p*np.log(p+1e-12),axis=1)
        return np.stack([loss,conf,ent],axis=1)
    return _safe_auc(feats(lf,pf),feats(lt,pt))


def _recovery_gain(model, split):
    """Maximum forget-loss recovery across a small intervention suite.

    This is deliberately a *stress-test* rather than a claim of irreversibility.
    Multiple low-budget interventions reduce dependence on one arbitrary attack
    budget while keeping the normal audit computationally bounded.
    """
    if not hasattr(model,"fit") or not hasattr(model,"clone"):
        return 0.0
    best=0.0
    for epochs, lr, seed in ((1,0.01,801),(3,0.015,808),(5,0.03,815)):
        recovered=clone_model(model)
        before=float(np.mean(per_sample_loss(recovered,split.X_forget,split.y_forget)))
        recovered.fit(split.X_forget,split.y_forget,epochs=epochs,lr=lr,batch_size=32,random_state=seed)
        after=float(np.mean(per_sample_loss(recovered,split.X_forget,split.y_forget)))
        best=max(best,max(0.0,before-after))
    return float(best)


def _representation_membership_auc(model, split):
    """AUC of a classifier trying to distinguish forget vs non-forget representations."""
    rep_f=_representation(model,split.X_forget)
    n=min(len(split.X_forget),len(split.X_test))
    if n < 4: return 0.5
    rep_n=_representation(model,split.X_test[:n])
    return _safe_auc(rep_f[:n],rep_n[:n])

def _representation_probe_auc(model, split):
    """Multi-layer representation leakage probe, reported as a security diagnostic."""
    n=min(len(split.X_forget),len(split.X_retain))
    if n < 4: return 0.5
    if not hasattr(model,'hidden_representation'):
        return _representation_membership_auc(model,split)
    values=[]
    for layer in (-1,-2):
        try:
            rf=model.hidden_representation(split.X_forget[:n],layer=layer)
            rr=model.hidden_representation(split.X_retain[:n],layer=layer)
            values.append(_safe_auc(rf,rr))
        except (TypeError,IndexError):
            continue
    return float(max(values, key=lambda x:abs(x-0.5))) if values else _representation_membership_auc(model,split)


def _model_metrics(model, split):
    return {
        'proba': _proba(model, split.X_forget),
        'auc': _membership_auc(model, split),
        'forget_loss': float(np.mean(per_sample_loss(model, split.X_forget, split.y_forget))),
        'rep': _representation(model, split.X_forget),
        'representation_probe_auc': _representation_probe_auc(model, split),
        'gain': _recovery_gain(model, split),
        'disc_profile': discontinuity_profile(model, split.X_forget, split.y_forget),
        'disc': discontinuity_score(discontinuity_profile(model, split.X_forget, split.y_forget)),
        'acc': evaluate(model, split.X_retain, split.y_retain)['accuracy'],
        'forget_acc': evaluate(model, split.X_forget, split.y_forget)['accuracy'],
    }


def _jsd_distance(pa, pb):
    vals=[]
    for a,b in zip(pa,pb):
        mm=np.clip((a+b)/2,1e-12,None)
        aa=np.clip(a,1e-12,None); bb=np.clip(b,1e-12,None)
        kl1=np.sum(np.where(a>0,a*np.log2(aa/mm),0)); kl2=np.sum(np.where(b>0,b*np.log2(bb/mm),0))
        vals.append(max(0.0,0.5*(kl1+kl2)))
    return float(np.mean(vals)) if vals else 0.0


def _pairwise_reference_distribution(metrics):
    """Build empirical null distances among independently trained retrains."""
    out={k:[] for k in ('behavioral','membership','representation','retain','irreversibility','representation_probe')}
    out['discontinuity']=[float(m['disc']) for m in metrics]
    for i in range(len(metrics)):
        for j in range(i+1,len(metrics)):
            a,b=metrics[i],metrics[j]
            out['behavioral'].append(_jsd_distance(a['proba'],b['proba']))
            out['membership'].append(0.5*abs(a['auc']-b['auc']) + 0.5*abs(a['forget_loss']-b['forget_loss'])/(abs(b['forget_loss'])+1e-3))
            out['representation'].append(float(np.clip(1-_linear_cka(a['rep'],b['rep']),0,1)))
            out['retain'].append(abs(a['acc']-b['acc']))
            out['irreversibility'].append(abs(a['gain']-b['gain']))
            out['representation_probe'].append(abs(a['representation_probe_auc']-b['representation_probe_auc']))
    return out


def _distance_to_reference_metrics(candidate, ref):
    return {
        'behavioral':_jsd_distance(candidate['proba'],ref['proba']),
        'membership':0.5*abs(candidate['auc']-ref['auc']) + 0.5*abs(candidate['forget_loss']-ref['forget_loss'])/(abs(ref['forget_loss'])+1e-3),
        'representation':float(np.clip(1-_linear_cka(candidate['rep'],ref['rep']),0,1)),
        'retain':abs(candidate['acc']-ref['acc']),
        'irreversibility':abs(candidate['gain']-ref['gain']),
        'representation_probe':abs(candidate['representation_probe_auc']-ref['representation_probe_auc']),
        'discontinuity':float(candidate['disc']),
    }

def _calibrated_score(distance, null_dist):
    """Reference-alignment score from an empirical upper-tail p-value.

    Distances are non-negative and smaller means *more* retrain-like.  The
    verifier asks how much of the retrain null is at least as far away as the
    candidate.  A candidate at or below the reference median gets a high score.

    Resolution fix: with n null values the smallest empirical p is 1/(n+1),
    which is >= 0.05 for n < 19, so an axis could never reject.  When the
    candidate lies beyond the whole null we therefore also compute a log-normal
    tail p-value (robust centre/scale, scale floor) and report the smaller of the
    two.  This only affects candidates outside the observed null range.
    """
    x=np.asarray(null_dist,float)
    if len(x)<3:
        return float(np.clip(1-distance,0,1))
    d=float(distance)
    upper=(np.sum(x >= d)+1)/(len(x)+1)
    if np.sum(x >= d) == 0:
        from scipy.stats import norm
        lx=np.log(np.clip(x,1e-9,None)); centre=float(np.median(lx))
        scale=max(1.4826*float(np.median(np.abs(lx-centre))), 0.35)
        z=(np.log(max(d,1e-9))-centre)/scale
        upper=min(upper, float(norm.sf(z)))
    return float(np.clip(upper,0,1))


def _two_sided_log_p(value, null_values, floor=1e-4, min_scale=0.35):
    """Two-sided robust log-normal p-value for a reference-independent statistic.

    Used for the discontinuity probe: an exact-input overlay is *too jumpy*
    (value >> null) and an adaptive ball overlay is *too flat* (value << null,
    constant output inside the ball), so both tails are rejection regions.
    """
    from scipy.stats import norm
    lx=np.log(np.clip(np.asarray(null_values,float),floor,None)); centre=float(np.median(lx))
    scale=max(1.4826*float(np.median(np.abs(lx-centre))), min_scale)
    z=(np.log(max(float(value),floor))-centre)/scale
    return float(np.clip(2*norm.sf(abs(z)),0,1))



def _calibrate_discontinuity_scales(candidate_profile, calibration_metrics):
    """Compute per-scale two-sided scores from calibration references only.

    Evaluation references are intentionally not accepted by this helper. They
    are used for candidate-to-reference distances elsewhere, not to fit the
    discontinuity null. The log-normal tail approximation remains a research
    heuristic and should be validated with larger ensembles before certification.
    """
    if not calibration_metrics:
        return {}
    return {
        scale: _two_sided_log_p(
            candidate_profile[scale],
            [m["disc_profile"][scale] for m in calibration_metrics],
        )
        for scale in candidate_profile
    }


def _distance_to_reference(model, ref, split):
    pm=_proba(model,split.X_forget); pg=_proba(ref,split.X_forget)
    vals=[]
    for a,b in zip(pm,pg):
        mm=(a+b)/2
        kl1=np.sum(np.where(a>0,a*np.log2(a/(mm+1e-12)),0))
        kl2=np.sum(np.where(b>0,b*np.log2(b/(mm+1e-12)),0))
        vals.append(max(0.0,0.5*(kl1+kl2)))
    behavioral=float(np.mean(vals)) if vals else 0.0
    auc_model=_membership_auc(model,split); auc_ref=_membership_auc(ref,split)
    membership=abs(auc_model-auc_ref)
    rep=1-_linear_cka(_representation(model,split.X_forget),_representation(ref,split.X_forget))
    retain=abs(evaluate(model,split.X_retain,split.y_retain)['accuracy']-
               evaluate(ref,split.X_retain,split.y_retain)['accuracy'])
    irreversibility=abs(_recovery_gain(model,split)-_recovery_gain(ref,split))
    return {'behavioral':behavioral,'membership':membership,'representation':float(np.clip(rep,0,1)),
            'retain':float(retain),'irreversibility':float(irreversibility)}


def run_advanced_audit(model, retrained_model, original_model, split, runtime_seconds: float,
                       baseline_runtime: float | None = None, reference_models=None,
                       reference_metrics=None, candidate_metrics=None,
                       decision_thresholds=None) -> AdvancedAuditResult:
    """Compute RENA-X axes; optionally calibrate them to an independent retrain ensemble."""
    gold_runtime=max(float(baseline_runtime or 0.0),1e-12)
    p_model=_parameter_vector(model); p_gold=_parameter_vector(retrained_model)
    parameter_distance=_dist(p_model,p_gold) if p_model is not None and p_gold is not None and len(p_model)==len(p_gold) else -1.0

    pm=_proba(model,split.X_forget); pg=_proba(retrained_model,split.X_forget)
    vals=[]
    for a,b in zip(pm,pg):
        mm=np.clip((a+b)/2,1e-12,None)
        aa=np.clip(a,1e-12,None); bb=np.clip(b,1e-12,None)
        kl1=np.sum(np.where(a>0,a*np.log2(aa/mm),0)); kl2=np.sum(np.where(b>0,b*np.log2(bb/mm),0))
        vals.append(max(0.0,0.5*(kl1+kl2)))
    jsd=float(np.mean(vals)) if vals else 0.0
    behavioral=float(np.clip(1-jsd,0,1))

    auc_model=_membership_auc(model,split); auc_gold=_membership_auc(retrained_model,split)
    membership=float(np.clip(1-abs(auc_model-auc_gold)/0.20,0,1))

    rm=_representation(model,split.X_forget); rg=_representation(retrained_model,split.X_forget)
    representation=float(_linear_cka(rm,rg))
    acc_model=evaluate(model,split.X_retain,split.y_retain)['accuracy']; acc_gold=evaluate(retrained_model,split.X_retain,split.y_retain)['accuracy']
    retain=float(np.clip(1-abs(acc_gold-acc_model)/0.20,0,1))
    gain_model=_recovery_gain(model,split); gain_gold=_recovery_gain(retrained_model,split)
    intervention_leakage=float(np.clip(max(0,gain_gold-gain_model)/(gain_gold+1e-12),0,1))
    irreversibility=float(1-intervention_leakage)
    efficiency=float(np.clip(gold_runtime/max(float(runtime_seconds),1e-12),0,1))

    raw_evidence_profile={'behavioral_equivalence':behavioral,'membership_indistinguishability':membership,'representation_neutrality':representation,'retain_consistency':retain,'irreversibility':irreversibility,'efficiency':efficiency}
    # Separate multi-layer representation leakage probe; it is diagnostic only.
    representation_probe_auc = _representation_probe_auc(model,split)
    calibration=None
    disc_prof=discontinuity_profile(model, split.X_forget, split.y_forget)
    # Uncalibrated fallback: honest smooth nets sit at ~0.4-0.5 loss-change per unit sigma, exact-input
    # overlays at >>1; map <=1.0 -> 1 and >=3.0 -> 0. The calibrated path uses reference nulls instead.
    discontinuity_consistency=float(np.clip(1-(discontinuity_score(disc_prof)-1.0)/2.0,0,1))
    # Forget-specific evidence is deliberately independent of the retrain-consistency
    # p-values.  A no-op can be statistically compatible with some broad witnesses,
    # but it has made zero progress toward the requested forget-set behavior.
    # Compute reference-ensemble metrics before deriving certification measures.
    if reference_models and reference_metrics is None:
        reference_metrics = [_model_metrics(ref, split) for ref in reference_models]
    original_forget_loss = float(np.mean(per_sample_loss(original_model, split.X_forget, split.y_forget)))
    candidate_forget_loss = float(np.mean(per_sample_loss(model, split.X_forget, split.y_forget)))
    reference_forget_losses = [float(m['forget_loss']) for m in (reference_metrics or [])]
    target_forget_loss = float(np.median(reference_forget_losses)) if reference_forget_losses else float(np.mean(per_sample_loss(retrained_model, split.X_forget, split.y_forget)))
    denom = target_forget_loss - original_forget_loss
    if abs(denom) < 1e-8:
        forget_progress = 1.0 if abs(candidate_forget_loss-target_forget_loss) < 1e-6 else 0.0
    else:
        forget_progress = float(np.clip((candidate_forget_loss-original_forget_loss)/denom, 0.0, 1.0))
    reference_retain_acc = [float(m['acc']) for m in (reference_metrics or [])]
    target_retain_acc = float(np.median(reference_retain_acc)) if reference_retain_acc else float(acc_gold)
    retain_utility_ratio = float(np.clip(acc_model/(target_retain_acc+1e-12),0.0,1.0))
    if reference_models and len(reference_models)>=4:
        reference_metrics=reference_metrics if reference_metrics is not None else [_model_metrics(ref,split) for ref in reference_models]
        candidate_metrics=candidate_metrics if candidate_metrics is not None else _model_metrics(model,split)
        # Strict calibration/evaluation separation: the empirical null is learned
        # from one subset of retrains and the candidate is compared against a
        # disjoint subset. This prevents the reference distribution used for
        # calibration from also defining the evaluation target.
        n_cal=max(3, int(np.ceil(len(reference_models)*0.6)))
        n_cal=min(n_cal, len(reference_models)-1) if len(reference_models)>3 else len(reference_models)
        cal_metrics=reference_metrics[:n_cal]
        eval_metrics=reference_metrics[n_cal:] if len(reference_metrics)>n_cal else reference_metrics
        null=_pairwise_reference_distribution(cal_metrics)
        distances=[]
        for ref_metrics in eval_metrics:
            distances.append(_distance_to_reference_metrics(candidate_metrics,ref_metrics))
        median_d={k:float(np.median([d[k] for d in distances])) for k in distances[0]}
        # Keep every null summary tied to the calibration subset. The held-out
        # evaluation references are used only for candidate-to-reference distances.
        null['discontinuity']=[float(m['disc']) for m in cal_metrics]
        scores={k:_calibrated_score(median_d[k],null[k]) for k in median_d}
        # Per-scale calibration (Bonferroni over scales) uses calibration references
        # only; held-out evaluation references must not leak into this null.
        _cp=candidate_metrics['disc_profile']
        _pp=_calibrate_discontinuity_scales(_cp, cal_metrics)
        scores['discontinuity']=float(min(1.0,len(_pp)*min(_pp.values())))
        calibration_disc_scales={str(k):v for k,v in _pp.items()}
        discontinuity_score_value=scores['discontinuity']
        behavioral=scores['behavioral']; membership=scores['membership']; representation=scores['representation']; retain=scores['retain']; irreversibility=scores['irreversibility']
        discontinuity_consistency=discontinuity_score_value
        calibration={'mode':'held_out_empirical_retrain_null','n_references':len(reference_models),'uncalibrated_evidence_profile':raw_evidence_profile,
                     'calibration_reference_count':len(cal_metrics),'evaluation_reference_count':len(eval_metrics),
                     'null_pair_count':len(next(iter(null.values()))),'median_distances':median_d,'scores':scores,
                     'raw_null_summary':{k:{'median':float(np.median(v)),'mad':float(np.median(np.abs(np.asarray(v)-np.median(v))))} for k,v in null.items()},
                     'representation_probe_auc':representation_probe_auc,
                     'representation_probe_reference_summary':{
                         'median':float(np.median([m['representation_probe_auc'] for m in eval_metrics])),
                         'mad':float(np.median(np.abs(np.asarray([m['representation_probe_auc'] for m in eval_metrics])-np.median([m['representation_probe_auc'] for m in eval_metrics]))))
                     },
                     'discontinuity_per_scale_p':calibration_disc_scales,
                     'separation_rule':'first 60% of references calibrate the empirical null; disjoint remaining references evaluate candidate distance'}

    # The representation probe is deliberately reported as a separate security
    # diagnostic. It is calibrated, but not silently folded into the six-axis
    # certification gate; this avoids pretending that CKA/alignment alone proves
    # representation-level non-leakage.
    correctness_axes=[behavioral,membership,representation,retain,irreversibility,discontinuity_consistency]
    correctness_min=float(min(correctness_axes))
    thresholds = {
        "certification_alpha": 0.10,
        "review_alpha": 0.05,
        "forget_pass": 0.40,
        "forget_review": 0.20,
        "retain_pass": 0.90,
        "retain_review": 0.80,
    }
    if decision_thresholds:
        thresholds.update({k: float(v) for k, v in decision_thresholds.items() if k in thresholds})
    certification_alpha = thresholds["certification_alpha"]
    review_alpha = thresholds["review_alpha"]
    if calibration:
        # The calibrated values are empirical upper-tail p-values: small values
        # mean the candidate is farther from the retrain null than expected.
        # Therefore certification is a non-rejection rule, not a demand that
        # every p-value exceed 0.90. This keeps the continuous alignment score
        # separate from the statistical decision.
        if correctness_min >= certification_alpha:
            certification_verdict = "PASS"
        elif correctness_min >= review_alpha:
            certification_verdict = "REVIEW"
        else:
            certification_verdict = "FAIL"
        calibration["certification_alpha"] = certification_alpha
        calibration["review_alpha"] = review_alpha
        # A certification decision now requires two logically different kinds of
        # evidence: (1) consistency with independently retrained references and
        # (2) actual progress toward forgetting.  This blocks the no-op loophole.
        calibration["forget_progress"] = forget_progress
        calibration["retain_utility_ratio"] = retain_utility_ratio
        calibration["forget_progress_thresholds"] = {"pass":thresholds["forget_pass"], "review":thresholds["forget_review"]}
        calibration["security_diagnostics"] = {
            "representation_probe_auc": representation_probe_auc,
            "recovery_gain": gain_model,
            "interpretation": "Diagnostics are adversarial/security evidence; they are not silently converted into proof of erasure."
        }
        # Bug fix: a correctness p-value below review_alpha is a rejection and must
        # stay FAIL; previously the gate logic below silently downgraded it to REVIEW.
        if (correctness_min < review_alpha or forget_progress < thresholds["forget_review"]
                or retain_utility_ratio < thresholds["retain_review"]):
            certification_verdict = "FAIL"
        elif correctness_min >= certification_alpha and forget_progress >= thresholds["forget_pass"] and retain_utility_ratio >= thresholds["retain_pass"]:
            certification_verdict = "PASS"
        else:
            certification_verdict = "REVIEW"
        calibration["decision_rule"] = (
            f"PASS requires every mandatory retrain-consistency p-value >= {certification_alpha:.3f}, "
            f"forget-progress >= {thresholds['forget_pass']:.3f}, and retain utility >= {thresholds['retain_pass']:.3f}; "
            f"REVIEW otherwise when forget-progress >= {thresholds['forget_review']:.3f} and retain utility >= {thresholds['retain_review']:.3f}; "
            f"FAIL if any correctness p-value < {review_alpha:.3f} or below either safety floor."
        )
        calibration["decision_thresholds"] = dict(thresholds)
        calibration["threshold_source"] = "development_calibration" if decision_thresholds else "engineering_default"
        calibration["certification_verdict"] = certification_verdict
    else:
        certification_verdict = "PASS" if correctness_min >= 0.70 else ("REVIEW" if correctness_min >= 0.45 else "FAIL")
    return AdvancedAuditResult(
        behavioral,membership,representation,retain,irreversibility,efficiency,
        correctness_min,correctness_min,parameter_distance,jsd,_representation_membership_auc(model,split),representation_probe_auc,forget_progress,retain_utility_ratio,intervention_leakage,
        float(max(0,acc_gold-acc_model)),float(runtime_seconds),gold_runtime,
        scoring_basis='reference-distribution calibrated' if calibration else 'single retrained reference',
        calibration=calibration, discontinuity_consistency=discontinuity_consistency, discontinuity_profile=disc_prof)

