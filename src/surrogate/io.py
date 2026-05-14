"""minimal io helpers (configs use types.save_config / load_config)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def save_pytree_npz(path: Path | str, tree: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    flat, _ = jax.tree_util.tree_flatten(tree)
    payload = {f"t{i}": np.asarray(x) for i, x in enumerate(flat)}
    meta = json.dumps([x.shape if hasattr(x, "shape") else () for x in flat])
    np.savez_compressed(p, meta=np.array(meta), **payload)


def load_pytree_like(path: Path | str, template: Any) -> Any:
    raw = np.load(path, allow_pickle=False)
    meta = json.loads(str(raw["meta"].item()))
    flat_t, treedef = jax.tree_util.tree_flatten(template)
    leaves = []
    for i, _ in enumerate(meta):
        leaves.append(jnp.asarray(raw[f"t{i}"], dtype=jnp.float32))
    if len(leaves) != len(flat_t):
        raise ValueError("checkpoint incompatible with template")
    return jax.tree_util.tree_unflatten(treedef, leaves)
