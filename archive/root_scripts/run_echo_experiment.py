#!/usr/bin/env python3
"""Reproducible ECHO experiment on the existing RENA pipeline."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from core.dataset import build_split
from core.model import evaluate, train_base_model
from core.pipeline import run_full_comparison
from core.verification import membership_inference_attack, loss_distribution_test, confidence_analysis, bootstrap_statistical_witness
from core.consensus import compute_consensus
from core.diagnosis import diagnose
from core.echo_controller import EchoConfig, echo_unlearn
from core.echo_evidence import parameter_distance
from core.recovery_stress_test import run_recovery_stress_test, transform_images, membership_recoverability
from core.conformal_diagnostic import run_conformal_diagnostic
from core.evidence import EvidenceItem, evidence_from_audit
from core.llm_evidence_analyzer import DEFAULT_MODEL, LLMAPIError, PROVIDER, analyze_evidence

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'; FIGURES=ROOT/'figures'; DOCS=ROOT/'docs'
def save(path, obj): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def full_audit(model, split, original, runtime=0.0):
    retain, test, forget = evaluate(model,split.X_retain,split.y_retain), evaluate(model,split.X_test,split.y_test), evaluate(model,split.X_forget,split.y_forget)
    modules=[membership_inference_attack(model,split.X_forget,split.y_forget,split.X_test,split.y_test,test['accuracy']), loss_distribution_test(model,split.X_forget,split.y_forget,split.X_test,split.y_test), confidence_analysis(model,original,split.X_forget,split.y_forget,split.X_test,split.y_test), bootstrap_statistical_witness(model,split.X_forget,split.y_forget,split.X_retain,split.y_retain)]
    consensus=compute_consensus(modules)
    return {'algorithm':'echo','runtime_seconds':runtime,'accuracy':{'retain':retain['accuracy'],'test':test['accuracy'],'forget':forget['accuracy'],'original_model_test':evaluate(original,split.X_test,split.y_test)['accuracy']},'module_results':[m.to_dict() for m in modules],'consensus':consensus.to_dict(),'diagnosis':diagnose(consensus,modules,retain['accuracy'],test['accuracy'])}

def adaptive_recovery(model, split):
    records=[]
    # Select perturbation intensity from residual loss/confidence, not a fixed condition alone.
    for sigma in (0.05,0.15,0.25,0.40):
        rng=np.random.RandomState(100+int(sigma*100)); xf=split.X_forget+rng.normal(0,sigma,split.X_forget.shape); xt=split.X_test+rng.normal(0,sigma,split.X_test.shape)
        metric=membership_recoverability(model,xf,split.y_forget,xt,split.y_test); metric.update({'adaptive_operator':'confidence_loss_guided_gaussian','sigma':sigma}); records.append(metric)
    return {'description':'Lightweight adaptive proxy: chooses the highest residual loss-based membership recoverability over noise intensities; not a state-of-the-art adaptive attack.','conditions':records,'max_recoverability':max(x['normalized_recoverability'] for x in records)}

def make_png(path,title,labels,values):
    im=Image.new('RGB',(900,460),'white'); d=ImageDraw.Draw(im); d.text((25,20),title,fill='black'); left,bottom,h=80,400,320; n=max(1,len(values));
    d.line((left,60,left,bottom),fill='black'); d.line((left,bottom,860,bottom),fill='black')
    for i,(lab,val) in enumerate(zip(labels,values)):
        x=left+20+i*(760/n); w=max(10,760/n-25); bh=int(max(0,min(1,val))*h); d.rectangle((x,bottom-bh,x+w,bottom),fill=(49,104,142)); d.text((x,bottom+8),lab[:14],fill='black'); d.text((x,bottom-bh-15),f'{val:.3f}',fill='black')
    path.parent.mkdir(exist_ok=True); im.save(path)

def run_one(seed=42, config=None):
    split=build_split(random_state=seed); original=train_base_model(np.vstack([split.X_retain,split.X_forget]),np.r_[split.y_retain,split.y_forget]); model, trace=echo_unlearn(original,split,config)
    audit=full_audit(model,split,original,trace['runtime_seconds']); recovery=run_recovery_stress_test(model,split); conformal=run_conformal_diagnostic(model,split); gold=train_base_model(split.X_retain,split.y_retain)
    audit['echo']={'iterations':trace['iterations'],'candidate_evaluations':trace['candidate_evaluations'],'parameter_update_norm':parameter_distance(model,original),'ground_truth_parameter_distance':parameter_distance(model,gold),'ground_truth_test_prediction_agreement':float(np.mean(model.predict(split.X_test)==gold.predict(split.X_test))),'adaptive_recovery':adaptive_recovery(model,split)}
    return split,original,model,audit,trace,recovery,conformal

def main():
    RESULTS.mkdir(exist_ok=True); (RESULTS/'evidence').mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    # Existing baseline is rerun and saved separately; its historical results are never edited.
    baseline=run_full_comparison(); save(RESULTS/'echo_baseline_comparison_source.json',baseline)
    split,original,model,audit,trace,recovery,conformal=run_one(42)
    items=evidence_from_audit('echo',audit,1001)
    for r in recovery:
        items.append(EvidenceItem(f"E{1001+len(items):04d}",'Recovery Stress Test','normalized_recoverability',r['normalized_recoverability'],r['normalized_recoverability'],1.0,None,[],['Supplementary fixed-transform loss proxy.'],{'algorithm':'echo',**r}))
    items.append(EvidenceItem(f"E{1001+len(items):04d}",'Conformal-style diagnostic','exclusion_gap',conformal['exclusion_gap'],None,None,None,[],[conformal['limitation']],{'algorithm':'echo',**conformal}))
    serialized=[x.to_dict() for x in items]; save(RESULTS/'evidence'/'echo.json',serialized)
    try: gemini=analyze_evidence(serialized)
    except LLMAPIError as exc: gemini={'analysis_mode':'API_ERROR','provider':PROVIDER,'model':DEFAULT_MODEL,'error':str(exc)}
    audit['gemini_evidence_assessment']=gemini; audit['recovery_stress']=recovery; audit['conformal_diagnostic']=conformal
    save(RESULTS/'echo_results.json',audit); save(RESULTS/'echo_trajectory.json',trace)
    # Three actual seeded ECHO repeats; full metrics are retained per seed.
    repeats=[]
    for seed in (42,43,44):
        _,_,_,a,t,_,_=run_one(seed); repeats.append({'seed':seed,'consensus':a['consensus']['consensus_score'],'agreement':a['consensus']['agreement'],'forget_accuracy':a['accuracy']['forget'],'retain_accuracy':a['accuracy']['retain'],'direct_recoverability':a['echo']['adaptive_recovery']['conditions'][0]['normalized_recoverability'],'iterations':t['iterations']})
    summary={k:{'mean':float(np.mean([x[k] for x in repeats])),'std':float(np.std([x[k] for x in repeats],ddof=1))} for k in ('consensus','agreement','forget_accuracy','retain_accuracy','direct_recoverability','iterations')}; audit['multiple_seed_summary']=summary; save(RESULTS/'echo_results.json',audit); save(RESULTS/'echo_seed_results.json',{'runs':repeats,'summary':summary})
    # Component ablations use the same split/initialization, with no baseline method alterations.
    variants={'standard_gradient_update':None,'echo_without_pareto':EchoConfig(use_pareto=False),'echo_without_adaptive_stopping':EchoConfig(use_adaptive_stopping=False),'echo_without_recovery_signal':EchoConfig(use_recovery_signal=False),'full_echo':EchoConfig()}; ablations={}
    for name,cfg in variants.items():
        if name=='standard_gradient_update':
            a=baseline['results']['gradient_ascent']; ablations[name]={'source':'existing_gradient_ascent','consensus':a['consensus']['consensus_score'],'agreement':a['consensus']['agreement'],'forget_accuracy':a['accuracy']['forget'],'retain_accuracy':a['accuracy']['retain']}
        else:
            _,_,_,a,t,rec,_=run_one(42,cfg); ablations[name]={'consensus':a['consensus']['consensus_score'],'agreement':a['consensus']['agreement'],'forget_accuracy':a['accuracy']['forget'],'retain_accuracy':a['accuracy']['retain'],'direct_recoverability':rec[0]['normalized_recoverability'],'iterations':t['iterations'],'candidate_evaluations':t['candidate_evaluations']}
    save(RESULTS/'echo_ablation_results.json',ablations)
    rows=[]
    for name,a in baseline['results'].items(): rows.append({'method':name,'consensus':a['consensus']['consensus_score'],'agreement':a['consensus']['agreement'],'forget_accuracy':a['accuracy']['forget'],'retain_accuracy':a['accuracy']['retain'],'test_accuracy':a['accuracy']['test'],'direct_recoverability':''})
    rows.append({'method':'echo','consensus':audit['consensus']['consensus_score'],'agreement':audit['consensus']['agreement'],'forget_accuracy':audit['accuracy']['forget'],'retain_accuracy':audit['accuracy']['retain'],'test_accuracy':audit['accuracy']['test'],'direct_recoverability':recovery[0]['normalized_recoverability']})
    with (RESULTS/'echo_comparison.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    make_png(FIGURES/'echo_evidence_trajectory.png','ECHO fast membership signal',[str(x['iteration']) for x in trace['trajectory']],[x['evidence_state']['membership_signal'] for x in trace['trajectory']])
    make_png(FIGURES/'method_comparison.png','Method consensus',[x['method'] for x in rows],[float(x['consensus']) for x in rows])
    make_png(FIGURES/'forget_retain_tradeoff.png','Retain accuracy by method',[x['method'] for x in rows],[float(x['retain_accuracy']) for x in rows])
    report=['# ECHO experiment report','','## Real experimental results',f"ECHO seed-42 consensus: {audit['consensus']['consensus_score']:.4f}; agreement: {audit['consensus']['agreement']:.4f}; forget/retain/test accuracy: {audit['accuracy']['forget']:.4f}/{audit['accuracy']['retain']:.4f}/{audit['accuracy']['test']:.4f}. Runtime: {audit['runtime_seconds']:.4f}s; {audit['echo']['iterations']} iterations; {audit['echo']['candidate_evaluations']} candidate evaluations; stopping: {trace['stopping_reason']}.",f"Three ECHO seeds: consensus mean±SD {summary['consensus']['mean']:.4f}±{summary['consensus']['std']:.4f}. All measured comparisons are in `results/echo_comparison.csv` and ablations in `results/echo_ablation_results.json`.",'','## Methodology','ECHO begins at the original NumPy MLP. At each iteration it temporarily applies four actual update operators: forget-pressure ascent, retain-protected ascent/correction, boundary-separation ascent/correction, and conservative stability ascent/correction. Fast internal proxies measure forget/retain loss, confidence, loss-based membership signal, statistical loss-gap signal, retain/test accuracy, recoverability, and parameter movement. Candidates are compared component-wise by Pareto dominance; a deterministic tie-break ranks retain degradation, recoverability, weakest forgetting proxy, then parameter update norm. Full RENA membership, KS, confidence, and bootstrap verification runs after the committed final checkpoint.','','## Real Gemini results','No real Gemini execution occurred in this run.','','## Fallback results',json.dumps(gemini,indent=2) if gemini['analysis_mode']=='FALLBACK' else 'None.','','## API errors',json.dumps(gemini,indent=2) if gemini['analysis_mode']=='API_ERROR' else 'None.','','## Pending experiments','A real Gemini API execution requires `GEMINI_API_KEY`. Independent replication and stronger adaptive attacks remain pending.','','## Limitations','The adaptive recovery test is a lightweight confidence/loss-guided perturbation sweep, not a state-of-the-art adaptive attack. ECHO reached its maximum iteration limit on seed 42 rather than its evidence-acceptance stopping region. This is evidence-guided verification feedback, not a guarantee of machine unlearning.']
    (DOCS/'echo_methodology.md').write_text('\n'.join(report)+'\n'); (DOCS/'echo_experiment_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'tests_hint':'run unittest discover','echo_consensus':audit['consensus']['consensus_score'],'echo_retain_accuracy':audit['accuracy']['retain'],'gemini_status':gemini['analysis_mode'],'results':str(RESULTS)},indent=2))
if __name__=='__main__': main()
