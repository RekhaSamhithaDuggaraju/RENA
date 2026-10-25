from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
 p=Path('results/research_suite.json'); data=json.loads(p.read_text()); out=Path('figures/research'); out.mkdir(parents=True,exist_ok=True)
 summary=data['summary']; names=[x['algorithm'] for x in summary]; means=[x['rena_x']['estimate'] for x in summary]; lows=[x['rena_x']['low'] for x in summary]; highs=[x['rena_x']['high'] for x in summary]
 plt.figure(figsize=(10,5)); plt.errorbar(names,means,yerr=[[m-l for m,l in zip(means,lows)],[h-m for h,m in zip(highs,means)]],fmt='o'); plt.xticks(rotation=35,ha='right'); plt.ylabel('RENA-X worst-case score'); plt.title('RENA-X method comparison with 95% bootstrap intervals'); plt.tight_layout(); plt.savefig(out/'rena_x_method_comparison.png',dpi=180); plt.close()
 axes=data['ranking'][0]['rena_x_axes']; keys=['behavioral_equivalence','membership_indistinguishability','representation_neutrality','retain_consistency','irreversibility','efficiency']; vals=[axes.get(k,0) for k in keys]
 plt.figure(figsize=(9,5)); plt.bar(keys,vals); plt.ylim(0,1); plt.xticks(rotation=35,ha='right'); plt.ylabel('Score'); plt.title(f"RENA-X audit profile: {data['ranking'][0]['algorithm']}"); plt.tight_layout(); plt.savefig(out/'rena_x_audit_profile.png',dpi=180); plt.close()
 seq=data['sequential_unlearning']; plt.figure(figsize=(7,5)); plt.plot([x['step'] for x in seq],[x['forget_loss'] for x in seq],marker='o'); plt.xlabel('Sequential forgetting step'); plt.ylabel('Forget-set loss'); plt.title('Sequential unlearning persistence trajectory'); plt.tight_layout(); plt.savefig(out/'sequential_unlearning.png',dpi=180); plt.close()
 print(f'Generated {len(list(out.glob("*.png")))} research figures in {out}')
if __name__=='__main__': main()
