#!/usr/bin/env python3
"""Immutable RENA-v2 calibration experiment runner.

Each invocation creates a new directory under results/rena_v2.  RRA and the
parameter proxy use only original/audited/retrained model states and held-out
test inputs; they never consume RENA witness or consensus values.
"""
from __future__ import annotations

import argparse, csv, json, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from core.checkpoints import save_model_state
from core.consensus import consensus_baselines
from core.dataset import build_split
from core.echo_controller import echo_unlearn
from core.model import evaluate, train_base_model
from core.prism_unlearning import prism_unlearn
from core.retrain_alignment import relative_retrain_alignment
from core.unlearning import fine_tuning_unlearn, gradient_ascent_unlearn, ground_truth_retrain, sisa_unlearn
from core.verification import bootstrap_statistical_witness, confidence_analysis, loss_distribution_test, membership_inference_attack

ROOT = Path(__file__).resolve().parent
METHODS = ("fine_tuning", "gradient_ascent", "ground_truth_retraining", "sisa", "prism", "echo")
SEEDS, FRACTIONS = tuple(range(42, 52)), (0.05, 0.10, 0.20, 0.30)
RRA_THRESHOLD = 0.5


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audited_model(method, split, original):
    if method == "fine_tuning": return fine_tuning_unlearn(original, split.X_retain, split.y_retain), {}
    if method == "gradient_ascent": return gradient_ascent_unlearn(original, split.X_forget, split.y_forget, split.X_retain, split.y_retain), {}
    if method == "ground_truth_retraining": return ground_truth_retrain(split.X_retain, split.y_retain), {}
    if method == "sisa":
        x, y = np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget]
        model, shards = sisa_unlearn(x, y, np.arange(len(split.X_retain), len(x)), np.arange(len(x)))
        return model, {"sisa_shards_retrained": shards}
    if method == "prism": return prism_unlearn(original, split)
    if method == "echo": return echo_unlearn(original, split)
    raise ValueError(method)


def parameter_alignment(original, audited, retrained):
    """Secondary independent proxy: relative L2 movement toward retraining."""
    def vector(model):
        if hasattr(model, "shard_models"):
            return np.concatenate([vector(child) for child in model.shard_models])
        return np.concatenate([p.ravel() for p in model.W + model.b])
    o, a, r = vector(original), vector(audited), vector(retrained)
    # SISA and non-SISA shapes may differ only if an unsupported method is added.
    if not (o.shape == a.shape == r.shape): return {"parameter_rra": None, "parameter_proxy_note": "incompatible parameter shapes"}
    ao, ar = float(np.linalg.norm(a-o)), float(np.linalg.norm(a-r))
    return {"parameter_rra": ao/(ao+ar+1e-12), "parameter_distance_audit_original": ao, "parameter_distance_audit_retrained": ar}


def run_case(method, seed, fraction, run_dir):
    split = build_split(forget_fraction=fraction, random_state=seed)
    x, y = np.vstack([split.X_retain, split.X_forget]), np.r_[split.y_retain, split.y_forget]
    original, retrained = train_base_model(x, y), ground_truth_retrain(split.X_retain, split.y_retain)
    started = time.perf_counter(); audited, trace = audited_model(method, split, original); runtime = time.perf_counter()-started
    test_accuracy = evaluate(audited, split.X_test, split.y_test)["accuracy"]
    witnesses = [membership_inference_attack(audited, split.X_forget, split.y_forget, split.X_test, split.y_test, test_accuracy), loss_distribution_test(audited, split.X_forget, split.y_forget, split.X_test, split.y_test), confidence_analysis(audited, original, split.X_forget, split.y_forget, split.X_test, split.y_test), bootstrap_statistical_witness(audited, split.X_forget, split.y_forget, split.X_retain, split.y_retain)]
    meta = {"method": method, "seed": seed, "forget_fraction": fraction, "dataset": "sklearn_digits", "run_id": run_dir.name}
    base = run_dir / "checkpoints" / f"method={method}" / f"seed={seed}_forget={fraction:.2f}"
    for role, model in (("original", original), ("audited", audited), ("ground_truth_retrained", retrained)):
        save_model_state(model, base.with_name(base.name + "_" + role), {**meta, "role": role, "architecture": type(model).__name__})
    return {"method": method, "seed": seed, "forget_fraction": fraction, "runtime_seconds": runtime,
            "accuracy": {k: evaluate(audited, X, yy)["accuracy"] for k, X, yy in (("retain",split.X_retain,split.y_retain),("forget",split.X_forget,split.y_forget),("test",split.X_test,split.y_test))},
            "module_results": [m.to_dict() for m in witnesses], "strategies": consensus_baselines(witnesses),
            "rra": relative_retrain_alignment(original, audited, retrained, split.X_test), "parameter_proxy": parameter_alignment(original, audited, retrained), "trace": trace}


def rows_from(runs):
    rows=[]
    for run in runs:
        row={"method":run["method"],"seed":run["seed"],"forget_fraction":run["forget_fraction"],"runtime_seconds":run["runtime_seconds"], **run["rra"], **run["parameter_proxy"]}
        row.update({f"{k}_accuracy":v for k,v in run["accuracy"].items()})
        for m in run["module_results"]: row["witness_"+m["module_name"]]=m["forget_score"]
        for name, value in run["strategies"].items():
            row[f"{name}_score"],row[f"{name}_agreement"],row[f"{name}_verdict"]=value["consensus_score"],value["agreement"],value["verdict"]
        row["BEST_INDIVIDUAL_score"]=max(m["forget_score"] for m in run["module_results"])
        rows.append(row)
    return rows


def spearman(x,y):
    result=stats.spearmanr(x,y); return {"rho":float(result.statistic),"p_value":float(result.pvalue),"n":len(x)}


def bootstrap(x,y,n=2000):
    rng=np.random.RandomState(20260907); values=[]
    for _ in range(n):
        i=rng.randint(0,len(x),len(x)); value=stats.spearmanr(np.asarray(x)[i],np.asarray(y)[i]).statistic
        if np.isfinite(value): values.append(value)
    return {"point_estimate":spearman(x,y)["rho"],"lower_ci":float(np.percentile(values,2.5)),"upper_ci":float(np.percentile(values,97.5)),"n_bootstrap_valid":len(values),"random_state":20260907}


def risk_curve(rows, strategy, thresholds):
    out=[]
    for agreement_threshold in thresholds:
        eligible=[]
        for r in rows:
            score, agreement=r[f"{strategy}_score"], r[f"{strategy}_agreement"]
            predicted = score >= .75
            # full RENA alone uses disagreement to abstain; comparators are forced.
            covered = (agreement >= agreement_threshold and (score >= .75 or score <= .35)) if strategy == "RENA_FULL" else (score >= .75 or score <= .35)
            if covered: eligible.append((predicted, r["rra"] >= RRA_THRESHOLD))
        error=float(np.mean([a != b for a,b in eligible])) if eligible else None
        out.append({"strategy":strategy,"agreement_threshold":agreement_threshold,"coverage":len(eligible)/len(rows),"covered_cases":len(eligible),"selective_risk":error})
    valid=[x for x in out if x["selective_risk"] is not None]
    ordered=sorted(valid,key=lambda x:x["coverage"])
    aurc=float(np.trapezoid([x["selective_risk"] for x in ordered],[x["coverage"] for x in ordered])) if len(ordered)>1 else None
    for x in out: x["aurc"]=aurc
    return out


def analyse(rows, run_dir):
    calibration, evaluation=[r for r in rows if r["seed"] <= 46],[r for r in rows if r["seed"] >= 47]
    # A smoke run has one seed only; it validates execution and is explicitly
    # exploratory, never a substitute for the held-out matrix evaluation.
    smoke_mode = not calibration or not evaluation
    if smoke_mode: calibration = evaluation = rows
    signals=["RENA_FULL_score","WEIGHTED_NO_OVERRIDE_score","NAIVE_AVERAGE_score","BEST_INDIVIDUAL_score"]+[k for k in rows[0] if k.startswith("witness_")]
    correlations={k:{**spearman([r[k] for r in evaluation],[r["rra"] for r in evaluation]),"bootstrap":bootstrap([r[k] for r in evaluation],[r["rra"] for r in evaluation])} for k in signals}
    # Thresholds are selected on calibration seeds only, then evaluated without retuning.
    threshold_grid=[.3,.4,.5,.6,.7,.8,.9]
    calibration_choice=[]
    for t in threshold_grid:
        c=risk_curve(calibration,"RENA_FULL",[t])[0]; calibration_choice.append(c)
    chosen=min((x for x in calibration_choice if x["selective_risk"] is not None), key=lambda x:(x["selective_risk"],-x["coverage"]))
    # Evaluation curves report all predeclared agreement gates; the selected
    # calibration gate is used only for the headline operating point below.
    risks=[]
    for strategy in ("RENA_FULL","WEIGHTED_NO_OVERRIDE","NAIVE_AVERAGE"):
        risks.extend(risk_curve(evaluation,strategy,threshold_grid if strategy=="RENA_FULL" else [0]))
    fp=[]
    for r in evaluation:
        predicted=r["RENA_FULL_score"] >= .75 and r["RENA_FULL_agreement"] >= chosen["agreement_threshold"]
        if predicted and r["rra"] < RRA_THRESHOLD: fp.append({k:r[k] for k in ("method","seed","forget_fraction","rra","RENA_FULL_score","RENA_FULL_agreement")})
    def counts(strategy):
        y=[r["rra"]>=RRA_THRESHOLD for r in evaluation]; p=[r[f"{strategy}_score"]>=.75 for r in evaluation]
        return {"false_positive_rate":sum(a and not b for a,b in zip(p,y))/max(1,sum(not b for b in y)),"false_negative_rate":sum(not a and b for a,b in zip(p,y))/max(1,sum(b for b in y)),"abstention_rate":sum(r[f"{strategy}_verdict"]=="Inconclusive" for r in evaluation)/len(evaluation)}
    contingency=[[sum(r["rra"]<.5 and r["RENA_FULL_score"]>=.75 for r in evaluation),sum(r["rra"]>=.5 and r["RENA_FULL_score"]>=.75 for r in evaluation)],[sum(r["rra"]<.5 and r["RENA_FULL_score"]<.75 for r in evaluation),sum(r["rra"]>=.5 and r["RENA_FULL_score"]<.75 for r in evaluation)]]
    statistical={"evaluation_n":len(evaluation),"correlations":correlations,"fisher_exact":{"table":contingency,**dict(zip(("odds_ratio","p_value"),map(float,stats.fisher_exact(contingency))))},"calibration_threshold_selection":{"held_out_evaluation":not smoke_mode,"smoke_mode":smoke_mode,"candidates":calibration_choice,"chosen":chosen}}
    dump(run_dir/"calibration_results.json", {"evaluation_n":len(evaluation),"correlations":correlations})
    dump(run_dir/"statistical_results.json", statistical)
    dump(run_dir/"risk_coverage_results.json", {"evaluation_n":len(evaluation),"curves":risks})
    dump(run_dir/"false_positive_analysis.json", {"rra_threshold":RRA_THRESHOLD,"cases":fp,"metrics":{s:counts(s) for s in ("RENA_FULL","WEIGHTED_NO_OVERRIDE","NAIVE_AVERAGE")}})
    return correlations, risks, fp, chosen


def figures(rows, correlations, risks, fp, run_dir):
    # Pillow is used deliberately: the archived Python build aborts inside
    # Matplotlib's native extension on this platform.
    from PIL import Image, ImageDraw
    f=ROOT/"figures"/"rena_v2"/run_dir.name; f.mkdir(parents=True,exist_ok=True)
    W,H=900,600
    def canvas(title, xlab, ylab):
        im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im); d.text((40,20),title,fill="black");d.line((90,H-80,W-40,H-80),fill="black",width=2);d.line((90,70,90,H-80),fill="black",width=2);d.text((W//2,H-45),xlab,fill="black");d.text((10,H//2),ylab,fill="black");return im,d
    def point(x,y): return (90+int(max(0,min(1,x))*(W-140)),H-80-int(max(0,min(1,y))*(H-150)))
    def save(im,name): im.save(f/(name+".png"))
    ev=[r for r in rows if r["seed"]>=47] or rows; im,d=canvas("Calibration relationship", "RENA audit score", "Independent RRA")
    for r in ev: d.ellipse((*point(r["RENA_FULL_score"],r["rra"]),*point(r["RENA_FULL_score"],r["rra"])),fill="steelblue")
    save(im,"calibration_reliability")
    keys=["RENA_FULL_score","NAIVE_AVERAGE_score","BEST_INDIVIDUAL_score"]; im,d=canvas("Correlation comparison", "Audit signal", "Spearman rho")
    for i,k in enumerate(keys):
        value=correlations[k]["rho"]; x=160+i*240; y=H-80-int((value+1)/2*(H-150)); d.rectangle((x,y,x+100,H-80),fill="steelblue");d.text((x,y-20),f"{value:.3f}",fill="black");d.text((x,H-70),k.replace("_score",""),fill="black")
    save(im,"correlation_comparison")
    im,d=canvas("Risk-coverage curve", "Coverage", "Selective risk")
    colors={"RENA_FULL":"steelblue","WEIGHTED_NO_OVERRIDE":"darkorange","NAIVE_AVERAGE":"seagreen"}
    for s,c in colors.items():
        pts=[point(x["coverage"],x["selective_risk"]) for x in risks if x["strategy"]==s and x["selective_risk"] is not None]
        if len(pts)>1:d.line(pts,fill=c,width=3)
        for p in pts:d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=c)
        d.text((600,80+25*list(colors).index(s)),s,fill=c)
    save(im,"risk_coverage_curve")
    im,d=canvas("False-positive cases", "Forget fraction", "RRA")
    if fp:
        for r in fp:
            p=point(r["forget_fraction"]/0.30,r["rra"]);d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill="crimson")
    else:d.text((350,300),"No false-positive cases",fill="black")
    save(im,"false_positive_heatmap")
    im,d=canvas("Threshold sensitivity", "Agreement threshold", "Selective risk")
    vals=[x for x in risks if x["strategy"]=="RENA_FULL" and x["selective_risk"] is not None];pts=[point((x["agreement_threshold"]-.3)/.6,x["selective_risk"]) for x in vals]
    if len(pts)>1:d.line(pts,fill="purple",width=3)
    for p in pts:d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill="purple")
    save(im,"threshold_sensitivity")


def main():
    p=argparse.ArgumentParser();p.add_argument("--quick",action="store_true");args=p.parse_args()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");run_dir=ROOT/"results"/"rena_v2"/(("smoke_" if args.quick else "matrix_")+stamp)
    run_dir.mkdir(parents=True,exist_ok=False); seeds,fractions=((42,),(.10,)) if args.quick else (SEEDS,FRACTIONS);runs=[]
    for seed in seeds:
        for fraction in fractions:
            for method in METHODS:
                print(f"running {method} seed={seed} forget={fraction:.2f}",flush=True);run=run_case(method,seed,fraction,run_dir);runs.append(run);dump(run_dir/"raw_runs"/f"{method}_seed{seed}_forget{fraction:.2f}.json",run)
    dump(run_dir/"raw_runs"/"all_runs.json",runs);rows=rows_from(runs)
    with (run_dir/"summary.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore")
        w.writeheader();w.writerows(rows)
    correlations,risks,fp,chosen=analyse(rows,run_dir);figures(rows,correlations,risks,fp,run_dir)
    dump(run_dir/"manifest.json",{"run_id":run_dir.name,"completed_conditions":len(runs),"expected_conditions":len(seeds)*len(fractions)*len(METHODS),"methods":METHODS,"seeds":seeds,"forget_fractions":fractions,"threshold_selection":chosen})
    print(json.dumps({"run_dir":str(run_dir),"completed_conditions":len(runs)},indent=2))
if __name__ == "__main__": main()
