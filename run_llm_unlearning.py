"""Run a controlled RENA LLM unlearning baseline and save its checkpoint/metadata.

Example (small local model recommended):
  python run_llm_unlearning.py --model <local-model-dir> --method gradient_ascent --output results/llm/ga
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
from core.llm.knowledge import build_synthetic_knowledge, knowledge_hash
from core.llm.unlearning import METHODS, TrainConfig, train_on_facts, unlearn_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='Base causal-LM model ID or local path')
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--output', required=True, help='Output directory for checkpoint and metadata')
    p.add_argument('--epochs', type=int, default=1, help='Unlearning/retraining epochs'); p.add_argument('--warmup-epochs', type=int, default=1, help='Fact-learning epochs before unlearning; ignored for retain-only retraining'); p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=256); p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install optional dependencies: python3 -m pip install -r requirements-llm.txt') from exc
    device=a.device
    if device=='auto': device='cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model).to(device)
    facts=build_synthetic_knowledge(); forget=[f for f in facts if f.split=='forget']; retain=[f for f in facts if f.split=='retain']
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); ckpt=out/'checkpoint'
    cfg=TrainConfig(a.method,a.epochs,a.learning_rate,a.max_length,a.beta,a.seed)
    warmup=[]
    if a.method != 'retain_only_retraining':
        warmup=train_on_facts(model,tokenizer,facts,epochs=a.warmup_epochs,learning_rate=a.learning_rate,max_length=a.max_length,seed=a.seed)
    result=unlearn_model(model,tokenizer,forget,retain,cfg,str(ckpt))
    result['forget_training_warmup']=warmup
    result['training_protocol']='retain-only from supplied initialization' if a.method == 'retain_only_retraining' else 'mixed forget+retain warmup followed by unlearning baseline'
    result.update({'base_model':a.model,'device':device,'knowledge_sha256':knowledge_hash(facts),'benchmark_note':'Run NIAH separately against this checkpoint and matched base/reference checkpoints.'})
    (out/'run_metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'method':result['method'],'checkpoint':result['checkpoint'],'epochs':len(result['history']),'metadata':str(out/'run_metadata.json')},indent=2))
if __name__=='__main__': main()
