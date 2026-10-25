"""Publication-oriented RENA research suite.

Adds reproducible experiments around the core audit: verifier ablations,
red-team evidence attacks, scenario/sequence/composition/persistence checks,
statistical uncertainty, ranking and machine-readable manifests.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib, json, math, platform, sys
from typing import Callable
import numpy as np
from sklearn.utils import shuffle
from core.dataset import build_split, UnlearningSplit
from core.model import train_base_model, evaluate, per_sample_loss, clone_model
from core.pipeline import run_audit, ALGORITHMS
from core.unlearning import ground_truth_retrain
from core.advanced_audit import run_advanced_audit, _model_metrics
from core.adversarial_lab import run_adversarial_suite
from core.benchmark_compare import compare_verifiers


def bootstrap_ci(values, statistic=np.mean, n_bootstrap=1000, seed=42):
    x=np.asarray(values,dtype=float)
    if len(x)==0: return {"estimate": None,"low":None,"high":None,"n":0}
    rng=np.random.RandomState(seed)
    samples=rng.choice(x,size=(n_bootstrap,len(x)),replace=True)
    stats=np.asarray([statistic(s) for s in samples])
    return {"estimate":float(statistic(x)),"low":float(np.quantile(stats,.025)),"high":float(np.quantile(stats,.975)),"n":int(len(x))}


def cohens_d(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2 or len(b)<2: return 0.0
    pooled=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b))/(pooled+1e-12))


def rank_methods(rows):
    """Rank methods with a transparent, bounded score; lower forget accuracy is better."""
    ranked=[]
    for r in rows:
        axes=r.get("rena_x_axes",{})
        score=float(r.get("rena_x_worst_case",0.0))
        ranked.append({**r,"ranking_score":score})
    return sorted(ranked,key=lambda x:(-x["ranking_score"],x.get("runtime_seconds",1e9)))


def _audit_axes(result):
    return result.get("diagnosis",{}).get("advanced_audit",{}).get("axes",{})


def verifier_ablation(result):
    """Leave-one-verifier-out consensus sensitivity without fabricating a rerun."""
    modules=result.get("module_results",[])
    scores=[float(m.get("forget_score",0)) for m in modules]
    if not scores: return []
    out=[]
    for i,m in enumerate(modules):
        rest=scores[:i]+scores[i+1:]
        mean=float(np.mean(rest)) if rest else 0.0
        spread=float(np.std(rest)) if rest else 1.0
        out.append({"removed":m["module_name"],"remaining_mean":mean,"remaining_std":spread,
                    "original_mean":float(np.mean(scores)),"delta":mean-float(np.mean(scores))})
    return out


def red_team_evidence(result, attack_strengths=(0.05,0.15,0.30,0.50)):
    """Stress-test the decision layer by corrupting one evidence score at a time."""
    modules=result.get("module_results",[])
    base=np.asarray([float(m.get("forget_score",0)) for m in modules])
    if len(base)<2: return []
    out=[]
    for strength in attack_strengths:
        for target,m in enumerate(modules):
            attacked=base.copy()
            attacked[target]=np.clip(attacked[target]-strength,0,1)
            agreement=float(1-np.std(attacked))
            mean=float(np.mean(attacked))
            out.append({"strength":strength,"target":m["module_name"],"mean_score":mean,
                        "agreement_proxy":agreement,"detected_as_suspicious":bool(mean<.55 or agreement<.75 or attacked[target]<.50)})
    return out



def adversarial_model_suite(seed=42, forget_fraction=0.10, reference_count=5):
    """Run model-level partial/selective/over/verifier-targeted controls."""
    split=build_split(forget_fraction,random_state=seed)
    X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
    original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
    refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=50000+seed*100+i) for i in range(reference_count)]
    return run_adversarial_suite(original,refs,split)


def verifier_comparison(result):
    """Compare calibrated RENA against the transparent ULTRA-style baseline."""
    audit=result.get("diagnosis",{}).get("advanced_audit",{})
    payload={"evidence_profile":audit.get("evidence_profile",{}),
             "certification_score":audit.get("certification_score",audit.get("worst_case_score",0.0)),
             "reference_calibration":result.get("reference_calibration",{})}
    return compare_verifiers(payload)


def partial_forgetting_curve(seed=42, forget_fraction=0.10, reference_count=10):
    """Evaluate 0/25/50/75/100% forgetting controls for monotonicity."""
    split=build_split(forget_fraction,random_state=seed)
    X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
    original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
    refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=50000+seed*100+i) for i in range(reference_count)]
    from core.adversarial_lab import partial_forgetting
    rows=[]
    for frac in (0.0,0.25,0.50,0.75,1.0):
        if frac==0.0:
            model=clone_model(original)
        elif frac==1.0:
            model=ground_truth_retrain(split.X_retain,split.y_retain,random_state=91000+seed)
        else:
            model=partial_forgetting(original,split,frac)
        audit=run_advanced_audit(model,refs[0],original,split,1.0,1.0,reference_models=refs)
        rows.append({"forgetting_fraction":frac,"rena_x":audit.worst_case_score,
                     "verdict":audit.calibration.get("certification_verdict") if audit.calibration else None,
                     "forget_accuracy":evaluate(model,split.X_forget,split.y_forget)["accuracy"],
                     "retain_accuracy":evaluate(model,split.X_retain,split.y_retain)["accuracy"]})
    scores=[r["rena_x"] for r in rows]
    # A verifier that behaves monotonically here should have a non-positive
    # Spearman trend if larger forgetting fraction means lower similarity.
    from scipy.stats import spearmanr
    rho=float(spearmanr([r["forgetting_fraction"] for r in rows],scores).statistic)
    return {"rows":rows,"spearman_fraction_vs_score":rho,
            "monotonicity_interpretation":"higher forgetting should generally reduce retrain-consistency score; this is a diagnostic, not a theorem"}


def verifier_benchmark_matrix(seed=42, forget_fraction=0.10, reference_count=10):
    """Run RENA-vs-ULTRA-style comparison plus partial-forgetting progression."""
    split=build_split(forget_fraction,random_state=seed)
    X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
    original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
    refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=50000+seed*100+i) for i in range(reference_count)]
    candidate=ground_truth_retrain(split.X_retain,split.y_retain,random_state=91000+seed)
    audit=run_advanced_audit(candidate,refs[0],original,split,1.0,1.0,reference_models=refs)
    result={"diagnosis":{"advanced_audit":{"evidence_profile":{
                "behavioral_equivalence":audit.behavioral_equivalence,
                "membership_indistinguishability":audit.membership_indistinguishability,
                "representation_neutrality":audit.representation_neutrality,
                "retain_consistency":audit.retain_consistency,
                "irreversibility":audit.irreversibility,
                "efficiency":audit.efficiency},
                "certification_score":audit.correctness_worst_case_score,
                "calibration":audit.calibration}},
            "reference_calibration":{"calibration":audit.calibration}}
    return {"verifier_comparison":compare_verifiers(result),"partial_forgetting_curve":partial_forgetting_curve(seed,forget_fraction,reference_count),
            "calibration":audit.calibration}

def multi_seed_calibration_benchmark(seeds=(42,43,44), forget_fraction=0.10, reference_count=8):
    """Run the calibrated verifier on independent split/model seeds.

    This is the next-level robustness check: calibration is repeated across
    genuinely different workloads rather than treating one Digits split as a
    universal operating point. Results are returned per seed plus aggregate
    mean/median, bootstrap CI, and abstention/certification rates.
    """
    rows=[]
    for seed in seeds:
        split=build_split(forget_fraction,random_state=seed)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
        refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=50000+seed*100+i) for i in range(reference_count)]
        candidate=ground_truth_retrain(split.X_retain,split.y_retain,random_state=91000+seed)
        audit=run_advanced_audit(candidate,refs[0],original,split,1.0,1.0,reference_models=refs)
        cal=audit.calibration or {}
        rows.append({
            'seed':int(seed),
            'rena_x':float(audit.worst_case_score),
            'certification_score':float(audit.correctness_worst_case_score),
            'verdict':cal.get('certification_verdict'),
            'representation_probe_auc':float(audit.representation_probe_auc),
            'calibration_reference_count':cal.get('calibration_reference_count'),
            'evaluation_reference_count':cal.get('evaluation_reference_count'),
            'null_pair_count':cal.get('null_pair_count'),
        })
    scores=[r['rena_x'] for r in rows]
    verdicts=[r['verdict'] for r in rows]
    return {
        'rows':rows,
        'summary':{
            'n_seeds':len(rows),
            'mean_rena_x':float(np.mean(scores)) if scores else None,
            'median_rena_x':float(np.median(scores)) if scores else None,
            'rena_x_bootstrap_ci':bootstrap_ci(scores,n_bootstrap=1000,seed=2026),
            'pass_rate':float(np.mean([v=='PASS' for v in verdicts])) if verdicts else None,
            'review_or_fail_rate':float(np.mean([v!='PASS' for v in verdicts])) if verdicts else None,
        },
        'interpretation':'Cross-seed robustness check for the calibrated audit; not evidence that ground-truth retraining is universally detectable or that the verifier dominates alternative auditors.'
    }


def calibration_control_discrimination(seeds=(42,43), forget_fraction=0.10, reference_count=8):
    """Compare known-good retraining with no-op and gradient-ascent controls.

    The purpose is calibration validation, not method ranking: a useful audit
    should accept independent retain-only retrains substantially more often
    than obvious retained-data or aggressive forgetting controls.
    """
    rows=[]
    for seed in seeds:
        split=build_split(forget_fraction,random_state=seed)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
        refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=50000+seed*100+i) for i in range(reference_count)]
        from core.adversarial_lab import over_unlearning
        controls={
            'retrain':ground_truth_retrain(split.X_retain,split.y_retain,random_state=91000+seed),
            'no_op':clone_model(original),
            'gradient_ascent':clone_model(original),
            'over_unlearning':over_unlearning(original,split),
        }
        for _ in range(8):
            controls['gradient_ascent'].gradient_step(split.X_forget,split.y_forget,lr=.03,ascent=True)
        for label,model in controls.items():
            audit=run_advanced_audit(model,refs[0],original,split,1.0,1.0,reference_models=refs)
            cal=audit.calibration or {}
            rows.append({'seed':int(seed),'control':label,'rena_x':float(audit.worst_case_score),
                         'verdict':cal.get('certification_verdict'),
                         'representation_probe_auc':float(audit.representation_probe_auc)})
    summary={}
    for control in sorted(set(r['control'] for r in rows)):
        subset=[r for r in rows if r['control']==control]
        summary[control]={'n':len(subset),'mean_rena_x':float(np.mean([r['rena_x'] for r in subset])),
                          'consistent_rate':float(np.mean([r['verdict']=='PASS' for r in subset]))}
    return {'rows':rows,'summary':summary,
            'interpretation':'Calibration discrimination should be read as a controlled sanity check. It is not a proof of security and does not establish superiority over external auditors.'}


def scenario_matrix(forget_fractions=(0.05,0.10,0.20)):
    """Run representative forget-request scenarios using real data splits."""
    scenarios=[]
    for frac in forget_fractions:
        split=build_split(frac,random_state=42)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y)
        for algo in ("ground_truth_retraining","fine_tuning","gradient_ascent","salun"):
            r=run_audit(algo,split,original,sample_ids=np.arange(len(X)))
            scenarios.append({"scenario":f"batch_delete_{frac:.2f}","forget_fraction":frac,"algorithm":algo,
                              "retain_accuracy":r["accuracy"]["retain"],"forget_accuracy":r["accuracy"]["forget"],
                              "rena_x_worst_case":r["diagnosis"]["advanced_audit"]["worst_case_score"]})
    return scenarios


def sequential_unlearning(seed=42, fractions=(0.05,0.05,0.05)):
    """Sequentially forget disjoint batches and record audit state after each step."""
    split=build_split(sum(fractions),random_state=seed)
    rng=np.random.RandomState(seed)
    idx=rng.permutation(len(split.X_forget))
    chunks=[]; start=0
    for f in fractions:
        n=max(1,int(round(f/sum(fractions)*len(idx))))
        chunks.append(idx[start:start+n]); start+=n
    model=train_base_model(np.vstack([split.X_retain,split.X_forget]),np.concatenate([split.y_retain,split.y_forget]))
    states=[]
    for step,chunk in enumerate(chunks,1):
        Xf=split.X_forget[chunk]; yf=split.y_forget[chunk]
        model.gradient_step(Xf,yf,lr=.03,ascent=True)
        for _ in range(4): model.gradient_step(split.X_retain,split.y_retain,lr=.015,ascent=False)
        states.append({"step":step,"forgotten_samples":int(sum(len(c) for c in chunks[:step])),
                       "forget_loss":float(np.mean(per_sample_loss(model,Xf,yf))),
                       "retain_accuracy":evaluate(model,split.X_retain,split.y_retain)["accuracy"]})
    return states


def composition_check(seed=42):
    """Compare sequential forgetting against one combined forget operation."""
    split=build_split(.20,random_state=seed)
    rng=np.random.RandomState(seed); idx=rng.permutation(len(split.X_forget)); a,b=idx[:len(idx)//2],idx[len(idx)//2:]
    full=train_base_model(np.vstack([split.X_retain,split.X_forget]),np.concatenate([split.y_retain,split.y_forget]))
    seq=clone_model(full)
    for chunk in (a,b):
        seq.gradient_step(split.X_forget[chunk],split.y_forget[chunk],lr=.03,ascent=True)
        seq.gradient_step(split.X_retain,split.y_retain,lr=.015,ascent=False)
    one=clone_model(full)
    one.gradient_step(split.X_forget,split.y_forget,lr=.03,ascent=True)
    one.gradient_step(split.X_retain,split.y_retain,lr=.015,ascent=False)
    diff=float(np.linalg.norm(seq.parameter_vector()-one.parameter_vector())/(np.linalg.norm(one.parameter_vector())+1e-12))
    return {"seed":seed,"relative_parameter_difference":diff,"interpretation":"Sequence and union are compared; nonzero difference indicates order/composition sensitivity."}


def persistence_check(seed=42):
    """Test whether forgetting survives later retain-only fine-tuning."""
    split=build_split(.10,random_state=seed)
    full=train_base_model(np.vstack([split.X_retain,split.X_forget]),np.concatenate([split.y_retain,split.y_forget]))
    model=clone_model(full)
    for _ in range(8): model.gradient_step(split.X_forget,split.y_forget,lr=.03,ascent=True)
    before=float(np.mean(per_sample_loss(model,split.X_forget,split.y_forget)))
    for _ in range(20): model.gradient_step(split.X_retain,split.y_retain,lr=.01,ascent=False)
    after=float(np.mean(per_sample_loss(model,split.X_forget,split.y_forget)))
    return {"before_forget_loss":before,"after_retain_finetune_forget_loss":after,"retention_ratio":float(after/(before+1e-12))}


def data_order_sensitivity(seed=42):
    split=build_split(.10,random_state=seed)
    X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
    original=train_base_model(X,y)
    ordered=run_audit("salun",split,original,sample_ids=np.arange(len(X)))
    Xs,ys=shuffle(X,y,random_state=seed+1)
    shuffled_model=train_base_model(Xs,ys)
    shuffled=run_audit("salun",split,shuffled_model,sample_ids=np.arange(len(X)))
    return {"ordered":ordered["diagnosis"]["advanced_audit"]["worst_case_score"],
            "shuffled":shuffled["diagnosis"]["advanced_audit"]["worst_case_score"],
            "absolute_delta":abs(ordered["diagnosis"]["advanced_audit"]["worst_case_score"]-shuffled["diagnosis"]["advanced_audit"]["worst_case_score"]) }


def experiment_manifest(config, result_paths=()):
    payload={"created_at":datetime.now(timezone.utc).isoformat(),"python":sys.version,
             "platform":platform.platform(),"numpy":np.__version__,"config":config,
             "result_paths":list(result_paths)}
    payload["manifest_id"]=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()[:16]
    return payload


def threshold_sensitivity(result, thresholds=(0.35, 0.45, 0.55, 0.65, 0.75)):
    """Evaluate how the observed module evidence changes under explicit score thresholds.

    This is a sensitivity analysis, not a replacement for the calibrated RENA defaults.
    """
    modules=result.get("module_results", [])
    if not modules:
        return []
    scores=np.asarray([float(m.get("forget_score",0.0)) for m in modules])
    agreement=float(result.get("consensus",{}).get("agreement",0.0))
    rows=[]
    for t in thresholds:
        mean=float(np.mean(scores))
        if agreement < 0.5:
            verdict="Inconclusive"
        elif mean >= t:
            verdict="Likely Forgotten"
        elif mean <= 1-t:
            verdict="Not Forgotten"
        else:
            verdict="Ambiguous"
        rows.append({"threshold":float(t),"mean_score":mean,"agreement":agreement,"verdict":verdict})
    return rows


def confidence_verdict(result):
    """Turn RENA-X score + consensus agreement into an evidence-strength label."""
    x=float(result.get("diagnosis",{}).get("advanced_audit",{}).get("worst_case_score",0.0))
    agreement=float(result.get("consensus",{}).get("agreement",0.0))
    confidence=float(np.clip(0.65*x+0.35*agreement,0,1))
    if confidence>=0.85 and x>=0.70: label="HIGH-CONFIDENCE"
    elif confidence>=0.65: label="MODERATE-CONFIDENCE"
    elif confidence>=0.45: label="LOW-CONFIDENCE / REVIEW"
    else: label="INSUFFICIENT EVIDENCE"
    return {"label":label,"confidence":confidence,"rena_x":x,"agreement":agreement,
            "abstained":bool(confidence<0.45 or agreement<0.35)}


def cost_quality_profile(runs):
    """Summarize the Pareto-style trade-off between audit quality and runtime."""
    rows=[]
    for r in runs:
        q=float(r.get("rena_x_worst_case",0.0)); rt=float(r.get("runtime_seconds",0.0))
        rows.append({**r,"quality_per_second":float(q/(rt+1e-9))})
    return sorted(rows,key=lambda x:(-x["rena_x_worst_case"],x["runtime_seconds"]))


def verifier_ground_truth_benchmark():
    """Controlled verifier benchmark with known synthetic states.

    Labels are deliberately constructed from evidence profiles; this validates the
    decision rule's ability to separate clear pass/fail/abstain cases, not real data erasure.
    """
    cases=[
        ("clean_forgetting",[.92,.90,.88,.94],"forgotten"),
        ("clear_retention",[.08,.12,.10,.06],"retained"),
        ("mixed_evidence",[.92,.20,.85,.15],"inconclusive"),
        ("borderline",[.68,.66,.70,.64],"inconclusive"),
        ("mostly_forgotten",[.84,.81,.78,.82],"forgotten"),
        ("mostly_retained",[.18,.22,.25,.15],"retained"),
    ]
    from core.verification import ModuleResult
    from core.consensus import compute_consensus
    rows=[]
    for name,scores,expected in cases:
        mods=[ModuleResult(f"Verifier-{i+1}",0.5,None,s,0.99,"synthetic benchmark",{"source":"controlled"}) for i,s in enumerate(scores)]
        actual=compute_consensus(mods).verdict
        mapped="forgotten" if actual=="Likely Forgotten" else "retained" if actual=="Not Forgotten" else "inconclusive"
        rows.append({"case":name,"expected":expected,"actual":mapped,"correct":mapped==expected,"scores":scores})
    accuracy=float(np.mean([r["correct"] for r in rows]))
    return {"accuracy":accuracy,"cases":rows,"note":"Controlled synthetic evidence profiles; not a real-world erasure benchmark."}


def consensus_ablation(result):
    """Compare RENA consensus against transparent averaging/override baselines."""
    from core.verification import ModuleResult
    from core.consensus import consensus_baselines
    mods=[]
    for m in result.get("module_results",[]):
        mods.append(ModuleResult(m["module_name"],float(m.get("raw_statistic",0.0)),m.get("p_value"),float(m["forget_score"]),float(m.get("reliability",1.0)),m.get("reliability_reason",""),m.get("detail",{})))
    return consensus_baselines(mods)


def red_team_summary(result):
    rows=red_team_evidence(result)
    return {"attacks":len(rows),"suspicious_rate":float(np.mean([r["detected_as_suspicious"] for r in rows])) if rows else 0.0,
            "rows":rows}



def _make_hard_forget_split(split, original_model, forget_fraction):
    """Select the most strongly memorized training points as the forget set.

    Selection uses only the original model and training data; it is fixed before
    any retrain/control is constructed. This creates a deliberately diagnostic
    membership-sensitive workload rather than relying on a lucky random split.
    """
    n=max(1,int(round(len(split.X_train)*forget_fraction)))
    losses=per_sample_loss(original_model,split.X_train,split.y_train)
    order=np.argsort(losses)[:n]
    mask=np.ones(len(split.X_train),dtype=bool); mask[order]=False
    return UnlearningSplit(
        X_train=split.X_train, y_train=split.y_train,
        X_retain=split.X_train[mask], y_retain=split.y_train[mask],
        X_forget=split.X_train[order], y_forget=split.y_train[order],
        X_test=split.X_test, y_test=split.y_test, scaler=split.scaler
    )

def _control_by_target_accuracy(original_model, split, target_accuracy, max_steps=80):
    """Create a measured partial-unlearning control near a target forget accuracy."""
    current=clone_model(original_model); best=clone_model(original_model)
    best_gap=abs(evaluate(best,split.X_forget,split.y_forget)['accuracy']-target_accuracy)
    for _ in range(max_steps):
        current.gradient_step(split.X_forget,split.y_forget,lr=0.08,ascent=True)
        acc=evaluate(current,split.X_forget,split.y_forget)['accuracy']
        gap=abs(acc-target_accuracy)
        if gap < best_gap:
            best=clone_model(current); best_gap=gap
        if acc <= target_accuracy:
            break
    return best

def _control_by_target_forget_loss(original_model, split, target_loss, max_steps=80):
    """Create a measured partial-unlearning control near a target mean forget loss."""
    current=clone_model(original_model); best=clone_model(original_model)
    best_gap=abs(float(np.mean(per_sample_loss(best,split.X_forget,split.y_forget)))-target_loss)
    for _ in range(max_steps):
        current.gradient_step(split.X_forget,split.y_forget,lr=0.08,ascent=True)
        loss=float(np.mean(per_sample_loss(current,split.X_forget,split.y_forget)))
        gap=abs(loss-target_loss)
        if gap < best_gap:
            best=clone_model(current); best_gap=gap
        if loss >= target_loss:
            break
    return best

def _over_unlearn_control(original_model, split, reference_models, max_steps=100):
    """Create an over-unlearned control after forget loss exceeds the retrain null."""
    gold_loss = float(np.median([np.mean(per_sample_loss(m, split.X_forget, split.y_forget)) for m in reference_models]))
    candidate = clone_model(original_model)
    for _ in range(max_steps):
        candidate.gradient_step(split.X_forget, split.y_forget, lr=0.20, ascent=True)
        loss = float(np.mean(per_sample_loss(candidate, split.X_forget, split.y_forget)))
        if loss > gold_loss * 1.50:
            break
    return candidate


def _audit_power_model(model, split, reference_models, original_model, reference_metrics=None, candidate_metrics=None):
    """Run only RENA-X for power analysis; the classical stack is a separate study."""
    result = run_advanced_audit(
        model=model,
        retrained_model=reference_models[0],
        original_model=original_model,
        split=split,
        runtime_seconds=1.0,
        baseline_runtime=1.0,
        reference_models=reference_models,
        reference_metrics=reference_metrics,
        candidate_metrics=candidate_metrics,
    )
    return result


def _auc_with_bootstrap(labels, scores, seed=42, n_bootstrap=1000):
    labels=np.asarray(labels,dtype=int); scores=np.asarray(scores,dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auc":None,"ci_low":None,"ci_high":None,"n":int(len(labels))}
    auc=float(__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(labels,scores))
    rng=np.random.RandomState(seed); vals=[]
    for _ in range(n_bootstrap):
        idx=rng.randint(0,len(labels),len(labels))
        if len(np.unique(labels[idx])) < 2: continue
        vals.append(float(__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(labels[idx],scores[idx])))
    if not vals:
        return {"auc":auc,"ci_low":None,"ci_high":None,"n":int(len(labels))}
    return {"auc":auc,"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),"n":int(len(labels))}


class _OutputPatchModel:
    """Deployment-time output overlay used as an adversarial bad control."""
    def __init__(self, base_model, X_patch, target_proba):
        import copy
        self.base_model=copy.deepcopy(base_model)
        self.X_patch=np.asarray(X_patch).copy()
        self.target_proba=np.asarray(target_proba).copy()
        self.n_classes=getattr(base_model,"n_classes",self.target_proba.shape[1])
        self._patch_enabled=True

    def _patch_indices(self,X):
        out=[]
        for x in np.asarray(X):
            hit=np.where(np.all(np.isclose(self.X_patch,x,rtol=1e-10,atol=1e-10),axis=1))[0]
            out.append(int(hit[0]) if len(hit) else -1)
        return out

    def predict_proba(self,X):
        probs=self.base_model.predict_proba(X).copy()
        if not self._patch_enabled: return probs
        for row,hit in enumerate(self._patch_indices(X)):
            if hit>=0: probs[row]=self.target_proba[hit]
        return probs

    def predict(self,X): return np.argmax(self.predict_proba(X),axis=1)
    def hidden_representation(self,X,layer=-1): return self.base_model.hidden_representation(X,layer=layer)
    def parameter_vector(self): return self.base_model.parameter_vector()
    def clone(self):
        import copy
        return copy.deepcopy(self)
    def fit(self,*args,**kwargs):
        self._patch_enabled=False
        self.base_model.fit(*args,**kwargs)
        return self


def _build_output_patch_controls(original_model, reference_models, split,
                                 suppressions=(0.01,0.02,0.03,0.04,0.05,0.06,0.07)):
    """Build seven exact-input output-patch attacks.

    Each attack overlays the forget-set outputs with the per-example median
    probability vector of the independent retrain ensemble, then suppresses
    the true-class probability by a small additional amount. The underlying
    weights and hidden representations remain unchanged.
    """
    ref_probs=np.stack([m.predict_proba(split.X_forget) for m in reference_models],axis=0)
    target=np.median(ref_probs,axis=0)
    target=target/np.maximum(target.sum(axis=1,keepdims=True),1e-12)
    y=split.y_forget; rows=np.arange(len(y))
    controls=[]
    for suppression in suppressions:
        patched=target.copy()
        true=patched[rows,y].copy()
        other=patched.copy(); other[rows,y]=0.0
        other_mass=np.maximum(other.sum(axis=1),1e-12)
        patched[rows,y]=true*(1.0-float(suppression))
        patched += other*(float(suppression)*true/other_mass)[:,None]
        controls.append((
            f"bad_output_patch_{int(round(suppression*100)):02d}pct",
            _OutputPatchModel(original_model,split.X_forget,patched),
            "output_patch",float(suppression)))
    return controls

class _RadiusPatchModel(_OutputPatchModel):
    """ADAPTIVE attacker: the overlay covers an RMS-radius ball around every forget input.

    Distance is the per-feature-std-normalised RMS distance, i.e. the same unit as
    the discontinuity probe's sigma, so a probe at sigma < radius cannot see the
    boundary.  This models an attacker who knows the probe family.
    """
    def __init__(self, base_model, X_patch, target_proba, radius, scale):
        super().__init__(base_model, X_patch, target_proba)
        self.radius=float(radius); self.scale=np.asarray(scale)

    def _patch_indices(self,X):
        Z=np.asarray(X)/self.scale; P=self.X_patch/self.scale
        d=np.sqrt(((Z[:,None,:]-P[None,:,:])**2).mean(axis=2))
        idx=d.argmin(axis=1); near=d[np.arange(len(Z)),idx]
        return [int(i) if n<=self.radius else -1 for i,n in zip(idx,near)]


def _build_radius_patch_controls(original_model, reference_models, split, radii=(0.15,0.40,0.80), suppression=0.03):
    base=_build_output_patch_controls(original_model,reference_models,split,suppressions=(suppression,))[0][1]
    scale=np.vstack([split.X_retain,split.X_forget]).std(axis=0)+1e-6
    return [(f"bad_radius_patch_r{int(round(r*100)):03d}",
             _RadiusPatchModel(original_model,split.X_forget,base.target_proba,r,scale),"radius_patch",float(r))
            for r in radii]


def verifier_power_study(seeds=(42, 43, 44), forget_fraction=0.10, n_retrains=10, hard_forget=False):
    """Distribution-calibrated verifier-power study with independent good models.

    For every seed, K independently initialized retain-only retrains form the
    empirical null. Each retrain is evaluated leave-one-out against the other
    K-1 retrains. Bad controls are separately constructed from the original model: no-op,
    measured partial unlearning at 25/50/75% of the retrain forget-loss gap,
    over-unlearning, and seven input-specific output-patch attacks. Labels are
    provenance-based and never derived from RENA-X scores.
    """
    rows=[]
    for seed in seeds:
        split=build_split(forget_fraction, random_state=seed)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
        if hard_forget:
            split=_make_hard_forget_split(split, original, forget_fraction)
        refs=[ground_truth_retrain(split.X_retain,split.y_retain,random_state=10000+seed*100+i) for i in range(n_retrains)]
        ref_metrics=[_model_metrics(m,split) for m in refs]
        ref_forget_accs=[m['forget_acc'] for m in ref_metrics]
        ref_forget_losses=[m['forget_loss'] for m in ref_metrics]
        original_forget_acc=evaluate(original,split.X_forget,split.y_forget)['accuracy']
        original_forget_loss=float(np.mean(per_sample_loss(original,split.X_forget,split.y_forget)))
        retrain_forget_acc=float(np.median(ref_forget_accs))
        retrain_forget_loss=float(np.median(ref_forget_losses))

        # Genuine good class: every retrain is held out from its calibration null.
        for i,candidate in enumerate(refs):
            loo=[m for j,m in enumerate(refs) if j!=i]
            loo_metrics=[m for j,m in enumerate(ref_metrics) if j!=i]
            a=_audit_power_model(candidate,split,loo,original,reference_metrics=loo_metrics,candidate_metrics=ref_metrics[i])
            rows.append({"seed":seed,"label":"good_retrained","algorithm":"independent_retrain",
                         "is_good":True,"reference_count":len(loo),"rena_x":a.worst_case_score,
                         "axes":a.to_dict(),"forget_accuracy":ref_forget_accs[i],
                         "forget_loss":ref_forget_losses[i],"attack_strength":None,"calibration":a.calibration})

        controls=[("bad_no_op",clone_model(original),"no_op",None)]
        # Accuracy is intentionally not used: hard-forget points can remain at
        # 100% accuracy while their loss changes substantially.
        loss_gap=retrain_forget_loss-original_forget_loss
        for frac in (0.25,0.50,0.75):
            target_loss=original_forget_loss+loss_gap*frac
            controls.append((f"bad_partial_{int(frac*100)}pct",
                             _control_by_target_forget_loss(original,split,target_loss),
                             "partial_unlearning",None))
        controls.append(("bad_over_unlearned",_over_unlearn_control(original,split,refs),"over_unlearning",None))
        controls.extend(_build_output_patch_controls(original,refs,split))
        controls.extend(_build_radius_patch_controls(original,refs,split))
        for label,model,algo,attack_strength in controls:
            a=_audit_power_model(model,split,refs,original,reference_metrics=ref_metrics,candidate_metrics=_model_metrics(model,split))
            rows.append({"seed":seed,"label":label,"algorithm":algo,"is_good":False,
                         "reference_count":len(refs),"rena_x":a.worst_case_score,"axes":a.to_dict(),
                         "forget_accuracy":evaluate(model,split.X_forget,split.y_forget)['accuracy'],
                         "forget_loss":float(np.mean(per_sample_loss(model,split.X_forget,split.y_forget))),
                         "attack_strength":attack_strength,"calibration":a.calibration})

    labels=[int(r['is_good']) for r in rows]
    scores=[float(r['rena_x']) for r in rows]
    auc=_auc_with_bootstrap(labels,scores)
    # Gate-aware score: this is deliberately reported separately from the
    # five-axis RENA-X alignment score because the certification decision also
    # requires forgetting progress and retain utility.
    gate_scores=[]; pass_flags=[]
    for r in rows:
        a=r["axes"]
        gate_scores.append(float(min(
            float(a.get("correctness_worst_case_score", 0.0))/0.10,
            float(a.get("forget_progress", 0.0))/0.40,
            float(a.get("retain_utility_ratio", 0.0))/0.90,
        )))
        pass_flags.append((a.get("calibration") or {}).get("certification_verdict") == "PASS")
    gate_auc=_auc_with_bootstrap(labels,gate_scores,seed=1907)
    good_n=sum(labels); bad_n=len(labels)-good_n
    tp=sum(int(p and y==1) for p,y in zip(pass_flags,labels))
    fp=sum(int(p and y==0) for p,y in zip(pass_flags,labels))
    gate_metrics={
        "strict_pass_tpr": float(tp/good_n) if good_n else None,
        "strict_pass_fpr": float(fp/bad_n) if bad_n else None,
        "n_good": int(good_n), "n_bad": int(bad_n),
        "note":"strict PASS is the complete certification decision, including forget-progress and retain-utility gates"
    }
    # Single-axis baselines and plain forget-loss MIA baseline.
    axis_aucs={}
    for axis in ('behavioral_equivalence','membership_indistinguishability','representation_neutrality','retain_consistency','irreversibility','discontinuity_consistency'):
        vals=[float(r['axes'][axis]) for r in rows]
        axis_aucs[axis]=_auc_with_bootstrap(labels,vals,seed=100+len(axis))
    # Same real forget-loss feature for every row, including GOOD retrains.
    plain_mia=[-float(r['forget_loss']) for r in rows]
    patch_rows=[r for r in rows if r.get('algorithm')=='output_patch']
    patch_accepts=[r for r in patch_rows if (r.get('calibration') or {}).get('certification_verdict')=='PASS']
    radius_rows=[r for r in rows if r.get('algorithm')=='radius_patch']
    radius_summary=[{'label':r['label'],'radius':r['attack_strength'],'verdict':(r.get('calibration') or {}).get('certification_verdict'),'discontinuity_p':r['axes']['discontinuity_consistency'],'forget_progress':r['axes']['forget_progress']} for r in radius_rows]
    nonadaptive=[r for r in rows if r.get('algorithm')!='radius_patch']
    auc_nonadaptive=_auc_with_bootstrap([int(r['is_good']) for r in nonadaptive],[float(r['rena_x']) for r in nonadaptive],seed=77)
    patch_budget_curve=[]
    for budget in range(1,len(patch_rows)+1):
        accepted=sum((r.get('calibration') or {}).get('certification_verdict')=='PASS' for r in patch_rows[:budget])
        patch_budget_curve.append({'budget':budget,'accepted':int(accepted),'false_accept_rate':float(accepted/budget)})
    return {"rows":rows,"n":len(rows),"n_retrains":n_retrains,
            "true_positive_rate":None,"true_negative_rate":None,"false_positive_rate":None,
            "roc_auc":auc,"single_axis_auc":axis_aucs,
            "plain_mia_auc":_auc_with_bootstrap(labels,plain_mia,seed=909),
            "plain_mia_bad_label_auc":_auc_with_bootstrap([1-y for y in labels],[-x for x in plain_mia],seed=910),
            "gate_aware_auc":gate_auc,
            "roc_auc_excluding_adaptive":auc_nonadaptive,
            "adaptive_radius_patch":{"results":radius_summary,"accepted":int(sum(x["verdict"]=="PASS" for x in radius_summary)),"threat_model":"adaptive overlay covering an RMS-radius ball around each forget input; defeats probes with sigma < radius","note":"Reported separately; included in roc_auc, excluded in roc_auc_excluding_adaptive."},
            "strict_pass_metrics":gate_metrics,
            "output_patch_attack":{
                "candidate_count":int(len(patch_rows)),
                "accepted_count":int(len(patch_accepts)),
                "false_accept_rate":float(len(patch_accepts)/len(patch_rows)) if patch_rows else None,
                "accepted_labels":[r['label'] for r in patch_accepts],
                "budget_curve":patch_budget_curve,
                "threat_model":"exact-input deployment-time output overlay using reference-median probabilities plus 1-7% true-class suppression; underlying weights and hidden representations remain unchanged"
            },
            "threshold_policy":"No fixed 0.70 threshold is used for the power claim; ROC/AUC is primary.",
            "hard_forget_set":bool(hard_forget),
            "note":"Good labels are independent retain-only retrains evaluated leave-one-out against the other retrains. Bad labels are constructed controls, including seven output-patch attacks. Default power study uses the ordinary random forget split; hard-forget mode is an optional diagnostic. This validates discrimination on this workload only; it is not an erasure certificate."}

def evidence_correlation(result):
    """Estimate correlation among classical verifier scores for one audit.

    A single audit has too few observations for a scientifically stable
    correlation estimate, so this returns a labelled *within-audit diagnostic*
    rather than pretending it is a population correlation study. Use the
    multi-seed experiment for publication statistics.
    """
    modules=result.get("module_results",[])
    names=[m.get("module_name","unknown") for m in modules]
    vals=np.asarray([float(m.get("forget_score",0.0)) for m in modules],dtype=float)
    if len(vals)<2:
        return {"status":"insufficient_evidence","matrix":{}}
    matrix=np.corrcoef(vals.reshape(1,-1))
    # One row cannot estimate pairwise correlations; explicitly expose that.
    return {"status":"not_estimable_from_single_audit","modules":names,
            "scores":vals.tolist(),
            "recommendation":"estimate witness correlations across many independently generated audits, not within one audit"}


def verifier_head_to_head_matrix(seed=42, forget_fraction=0.10, reference_count=10,
                                  attack_budget=12):
    """Evaluate RENA-X against simpler verifiers on clean and adversarial controls.

    The candidate labels are experimental labels supplied by the benchmark
    designer: exact retain-only retraining is GOOD; every deliberately
    compromised control is BAD. This lets us report FPR/FNR without letting any
    verifier define its own ground truth.
    """
    from core.adversarial_lab import (
        partial_forgetting, selective_forgetting, over_unlearning,
        verifier_targeted_deception,
    )
    from core.verifier_attack import rena_x_targeted_deception
    from core.verifier_matrix import evaluate_candidate, summarize_matrix

    split = build_split(forget_fraction, random_state=seed)
    X = np.vstack([split.X_retain, split.X_forget])
    y = np.concatenate([split.y_retain, split.y_forget])
    original = train_base_model(X, y, random_state=7000 + seed, fit_random_state=7000 + seed)
    refs = [ground_truth_retrain(split.X_retain, split.y_retain,
                                 random_state=50000 + seed * 100 + i)
            for i in range(reference_count)]
    ref_metrics = [_model_metrics(m, split) for m in refs]

    adaptive = rena_x_targeted_deception(
        original, refs, split, reference_metrics=ref_metrics, budget=attack_budget, seed=seed,
    )
    candidates = {
        "retrain": refs[0],
        "no_op": clone_model(original),
        "partial_25": partial_forgetting(original, split, 0.25),
        "partial_50": partial_forgetting(original, split, 0.50),
        "partial_75": partial_forgetting(original, split, 0.75),
        "selective": selective_forgetting(original, split),
        "over_unlearning": over_unlearning(original, split),
        "verifier_targeted": verifier_targeted_deception(original, refs[0], split),
        "rena_x_targeted": adaptive["candidate_model"],
    }
    labels = {name: (name == "retrain") for name in candidates}
    rows = [evaluate_candidate(name, model, original, split, refs, ref_metrics)
            for name, model in candidates.items()]
    summary = summarize_matrix(rows, labels)
    return {
        "dataset_summary": split.summary(),
        "seed": int(seed),
        "reference_count": int(reference_count),
        "candidate_labels": labels,
        "rows": rows,
        "summary": summary,
        "adaptive_attack": {k: v for k, v in adaptive.items() if k != "candidate_model"},
        "interpretation": (
            "This is a verifier stress test, not a claim that the controls are "
            "real-world attacks. The key result is whether an adaptive candidate "
            "can preserve residual forget information while obtaining acceptance "
            "from the complete verifier."
        ),
    }
