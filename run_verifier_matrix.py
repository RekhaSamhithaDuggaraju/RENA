#!/usr/bin/env python3
"""Run the v11 head-to-head verifier stress test."""
from __future__ import annotations
import argparse, json, time
from core.research_suite import verifier_head_to_head_matrix

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,default=42)
p.add_argument('--forget-fraction',type=float,default=.10)
p.add_argument('--reference-count',type=int,default=10)
p.add_argument('--attack-budget',type=int,default=12)
p.add_argument('--output',default='results/verifier_head_to_head.json')
a=p.parse_args()
t=time.time()
r=verifier_head_to_head_matrix(a.seed,a.forget_fraction,a.reference_count,a.attack_budget)
r['runtime_seconds']=time.time()-t
with open(a.output,'w') as f: json.dump(r,f,indent=2,default=float)
print(json.dumps({'output':a.output,'runtime_seconds':r['runtime_seconds'],'summary':r['summary'],'adaptive_selected':r['adaptive_attack']['selected']},indent=2))
