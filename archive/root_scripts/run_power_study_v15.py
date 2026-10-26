"""Reproduce the V15 verifier-power study (seeds 42/43/44, K=10 independent retrains per seed).

    PYTHONPATH=. python run_power_study_v15.py --seeds 42 43 44 --out results/verifier_power_study_v15.json

Seed 42 was the development seed for the V15 discontinuity axis; 43 and 44 are held out.
"""
import argparse, json, warnings
from core.research_suite import verifier_power_study
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
ap.add_argument("--n-retrains", type=int, default=10)
ap.add_argument("--out", default="results/verifier_power_study_v15.json")
a = ap.parse_args()
r = verifier_power_study(seeds=tuple(a.seeds), n_retrains=a.n_retrains)
json.dump(r, open(a.out, "w"), default=float, indent=1)
print("n", r["n"], "AUC", r["roc_auc"], "\nstrict", r["strict_pass_metrics"],
      "\nexcl-adaptive", r["roc_auc_excluding_adaptive"],
      "\noutput_patch accepted", r["output_patch_attack"]["accepted_count"], "/", r["output_patch_attack"]["candidate_count"],
      "\nradius accepted", r["adaptive_radius_patch"]["accepted"])
