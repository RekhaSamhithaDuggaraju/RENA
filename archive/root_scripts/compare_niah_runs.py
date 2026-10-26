"""Compare a candidate NIAH result JSON with a matched retain-only reference."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.llm.paired_niah import compare_niah_to_reference


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True); p.add_argument('--reference',required=True)
    p.add_argument('--candidate-name',default='candidate'); p.add_argument('--reference-name',default='retain_only_retraining')
    p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap-iterations',type=int,default=2000); a=p.parse_args()
    left=json.loads(Path(a.candidate).read_text(encoding='utf-8')); right=json.loads(Path(a.reference).read_text(encoding='utf-8'))
    result=compare_niah_to_reference(left,right,candidate_name=a.candidate_name,reference_name=a.reference_name,seed=a.seed,bootstrap_iterations=a.bootstrap_iterations)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); print(f'Saved comparison to {out}')

if __name__=='__main__': main()
