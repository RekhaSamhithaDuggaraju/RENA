from __future__ import annotations
import argparse,json
from pathlib import Path
from core.pipeline import run_full_comparison, ALGORITHMS
from core.research_suite import verifier_ablation, red_team_evidence, sequential_unlearning, composition_check, persistence_check, data_order_sensitivity, scenario_matrix, rank_methods, experiment_manifest, bootstrap_ci, cohens_d, threshold_sensitivity, confidence_verdict, consensus_ablation, verifier_ground_truth_benchmark, cost_quality_profile, verifier_power_study

def main():
    ap=argparse.ArgumentParser(description='RENA publication research suite')
    ap.add_argument('--seeds',nargs='+',type=int,default=[7,42,2026])
    ap.add_argument('--output',default='results/research_suite.json')
    args=ap.parse_args()
    runs=[]; ablations=[]; redteams=[]
    for seed in args.seeds:
        c=run_full_comparison(.10,seed)
        for algo in ALGORITHMS:
            r=c['results'][algo]
            runs.append({'seed':seed,'algorithm':algo,'rena_x_worst_case':r['diagnosis']['advanced_audit']['worst_case_score'],
                         'retain_accuracy':r['accuracy']['retain'],'forget_accuracy':r['accuracy']['forget'],'test_accuracy':r['accuracy']['test'],
                         'runtime_seconds':r['runtime_seconds'],'rena_x_axes':r['diagnosis']['advanced_audit']['axes']})
            if seed==args.seeds[0]:
                ablations.append({'algorithm':algo,'ablations':verifier_ablation(r),'red_team':red_team_evidence(r)})
    summary=[]
    for algo in ALGORITHMS:
        rows=[x for x in runs if x['algorithm']==algo]
        summary.append({'algorithm':algo,'rena_x':bootstrap_ci([x['rena_x_worst_case'] for x in rows]),
                        'retain_accuracy':bootstrap_ci([x['retain_accuracy'] for x in rows]),
                        'forget_accuracy':bootstrap_ci([x['forget_accuracy'] for x in rows]),
                        'runtime':bootstrap_ci([x['runtime_seconds'] for x in rows])})
    first=run_full_comparison(.10,args.seeds[0])
    payload={'manifest':experiment_manifest({'seeds':args.seeds,'forget_fraction':.10},['results/research_suite.json']),
             'runs':runs,'summary':summary,'ranking':rank_methods(runs),'verifier_ablation':ablations,
             'scenarios':scenario_matrix(),'sequential_unlearning':sequential_unlearning(),
             'composition_check':composition_check(),'persistence_check':persistence_check(),
             'data_order_sensitivity':data_order_sensitivity(),
             'red_team_summary':{'total_cases':sum(len(x['red_team']) for x in ablations)},
             'extended':{
                 'threshold_sensitivity': {a:threshold_sensitivity(first['results'][a]) for a in ALGORITHMS},
                 'confidence': {a:confidence_verdict(first['results'][a]) for a in ALGORITHMS},
                 'consensus_ablation': {a:consensus_ablation(first['results'][a]) for a in ALGORITHMS},
                 'controlled_synthetic_benchmark': verifier_ground_truth_benchmark(),
                 'verifier_power_study': verifier_power_study(seeds=args.seeds, forget_fraction=.10),
                 'cost_quality': cost_quality_profile(runs)}}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,default=float))
    print(f'Wrote research suite to {out}')
if __name__=='__main__': main()
