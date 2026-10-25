"""Population-level dependence analysis for RENA's classical witnesses."""
from __future__ import annotations
import numpy as np
from core.pipeline import run_audit
from core.dataset import build_split
from core.model import train_base_model


def witness_dependence(seeds=(7,42,2026), forget_fraction=.1, algorithm='gradient_ascent'):
    rows=[]; names=None
    for seed in seeds:
        split=build_split(forget_fraction,random_state=seed)
        X=np.vstack([split.X_retain,split.X_forget]); y=np.concatenate([split.y_retain,split.y_forget])
        original=train_base_model(X,y,random_state=7000+seed,fit_random_state=7000+seed)
        result=run_audit(algorithm,split,original,sample_ids=np.arange(len(X)),reference_count=0,split_seed=seed)
        mods=result.get('module_results',[])
        names=[m['module_name'] for m in mods]
        rows.append([float(m.get('forget_score',0)) for m in mods])
    arr=np.asarray(rows,float)
    corr=np.corrcoef(arr,rowvar=False) if len(rows)>=3 and arr.shape[1]>=2 else np.eye(arr.shape[1])
    return {'seeds':list(seeds),'algorithm':algorithm,'modules':names,'scores':arr.tolist(),
            'correlation_matrix':corr.tolist(),
            'note':'Population correlation across independent audit runs; reliability weights are not a substitute for dependence correction.'}
