"""Run the RENA modern-method comparison across multiple split seeds.

This is intentionally a CLI experiment runner: it writes raw per-run results and
aggregate mean/std statistics so the paper can report variability rather than one lucky seed.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from core.pipeline import run_full_comparison, ALGORITHMS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[7,42,2026])
    ap.add_argument('--forget-fraction', type=float, default=0.1)
    ap.add_argument('--output', default='results/modern_multiseed_benchmark.json')
    args=ap.parse_args()
    runs=[]
    for seed in args.seeds:
        comparison=run_full_comparison(forget_fraction=args.forget_fraction, random_state=seed)
        for algo in ALGORITHMS:
            r=comparison['results'][algo]
            runs.append({'seed':seed,'algorithm':algo,'consensus_score':r['consensus']['consensus_score'],
                         'agreement':r['consensus']['agreement'],'retain_accuracy':r['accuracy']['retain'],
                         'forget_accuracy':r['accuracy']['forget'],'test_accuracy':r['accuracy']['test'],
                         'runtime_seconds':r['runtime_seconds'],
                         'rena_x_worst_case':r['diagnosis']['advanced_audit']['worst_case_score']})
    summary={}
    for algo in ALGORITHMS:
        rows=[r for r in runs if r['algorithm']==algo]
        summary[algo]={}
        for key in ('consensus_score','agreement','retain_accuracy','forget_accuracy','test_accuracy','runtime_seconds','rena_x_worst_case'):
            vals=[float(r[key]) for r in rows]
            summary[algo][key]={'mean':statistics.mean(vals),'std':statistics.stdev(vals) if len(vals)>1 else 0.0}
    payload={'config':{'seeds':args.seeds,'forget_fraction':args.forget_fraction},'runs':runs,'summary':summary}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2))
    print(f'Wrote {len(runs)} runs to {out}')

if __name__=='__main__': main()
