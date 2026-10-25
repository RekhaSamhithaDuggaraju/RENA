"""Canonical experiment provenance for reproducible RENA audits."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys
import numpy as np


def _hash_array(x: np.ndarray) -> str:
    a=np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes()).hexdigest()


def git_revision(root: str = ".") -> str | None:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_manifest(split, *, algorithm: str, split_seed: int, reference_seed: int,
                   reference_retrains: int, model_seed: int = 42, method_seed: int | None = None) -> dict:
    payload={
        "dataset": "sklearn_digits",
        "dataset_split_hash": _hash_array(np.concatenate([split.X_train.ravel(),split.y_train.astype(float).ravel()])),
        "forget_set_hash": _hash_array(np.concatenate([split.X_forget.ravel(),split.y_forget.astype(float).ravel()])),
        "retain_set_hash": _hash_array(np.concatenate([split.X_retain.ravel(),split.y_retain.astype(float).ravel()])),
        "test_set_hash": _hash_array(np.concatenate([split.X_test.ravel(),split.y_test.astype(float).ravel()])),
        "model": {"architecture":"SimpleMLP(32,16)","initialization_seed":model_seed},
        "algorithm": algorithm,
        "split_seed": int(split_seed),
        "method_seed": method_seed,
        "reference": {"count":int(reference_retrains),"seed_base":int(reference_seed),"training":"retain_only_independent_retrain"},
        "software": {"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.platform(),"git_commit":git_revision()},
    }
    payload["manifest_hash"]=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    return payload
