"""Safe, reproducible NumPy checkpoint serialization for RENA models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from core.nn import SimpleMLP
from core.sisa import SISAEnsemble


def save_model_state(model, path: str | Path, metadata: dict) -> tuple[Path, Path]:
    """Save a SimpleMLP as ``.npz`` plus JSON metadata; reject opaque models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append rather than replace a suffix: condition identifiers legitimately
    # contain decimal forget fractions (for example ``forget=0.05``).
    npz_path = Path(str(path) + ".npz")
    json_path = Path(str(path) + ".json")
    if isinstance(model, SimpleMLP):
        arrays = {"model_kind": np.asarray("SimpleMLP"), "sizes": np.asarray(model.sizes), "n_classes": np.asarray(model.n_classes)}
        arrays.update({f"W_{i}": array for i, array in enumerate(model.W)})
        arrays.update({f"b_{i}": array for i, array in enumerate(model.b)})
    elif isinstance(model, SISAEnsemble):
        arrays = {"model_kind": np.asarray("SISAEnsemble"), "n_shards": np.asarray(model.n_shards)}
        for shard, child in enumerate(model.shard_models):
            arrays[f"sizes_{shard}"] = np.asarray(child.sizes)
            arrays[f"ids_{shard}"] = model.shard_indices[shard]
            for layer, value in enumerate(child.W): arrays[f"W_{shard}_{layer}"] = value
            for layer, value in enumerate(child.b): arrays[f"b_{shard}_{layer}"] = value
    else:
        raise TypeError(f"unsupported checkpoint model type: {type(model).__name__}")
    np.savez_compressed(npz_path, **arrays)
    payload = {**metadata, "format": "rena-simplemlp-npz-v1", "created_at_utc": datetime.now(timezone.utc).isoformat()}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return npz_path, json_path


def load_model_state(path: str | Path) -> SimpleMLP:
    path = Path(path)
    if path.suffix != ".npz":
        path = Path(str(path) + ".npz")
    with np.load(path, allow_pickle=False) as state:
        kind = str(state["model_kind"])
        if kind == "SISAEnsemble":
            ensemble = SISAEnsemble(int(state["n_shards"]))
            for shard in range(ensemble.n_shards):
                sizes = [int(x) for x in state[f"sizes_{shard}"]]
                child = SimpleMLP(sizes[0], sizes[-1], hidden_sizes=tuple(sizes[1:-1]), random_state=0)
                child.W = [state[f"W_{shard}_{i}"].copy() for i in range(len(sizes) - 1)]
                child.b = [state[f"b_{shard}_{i}"].copy() for i in range(len(sizes) - 1)]
                ensemble.shard_models.append(child); ensemble.shard_indices.append(state[f"ids_{shard}"].copy())
            return ensemble
        sizes = [int(x) for x in state["sizes"]]
        model = SimpleMLP(sizes[0], int(state["n_classes"]), hidden_sizes=tuple(sizes[1:-1]), random_state=0)
        model.W = [state[f"W_{i}"].copy() for i in range(len(sizes) - 1)]
        model.b = [state[f"b_{i}"].copy() for i in range(len(sizes) - 1)]
    return model
