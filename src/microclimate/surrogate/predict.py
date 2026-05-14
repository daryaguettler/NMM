"""rasterize trained mlp onto a problem grid."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from microclimate.config import GridConfig, ProblemConfig
from microclimate.grid import building_mask, make_grid
from microclimate.surrogate.model import forward_batch
from microclimate.types import TemperatureField


def load_surrogate_bundle(artifact_dir: Path) -> tuple[object, dict]:
    d = Path(artifact_dir).expanduser().resolve()
    pkl = d / "surrogate_params.pkl"
    meta_path = d / "surrogate_meta.json"
    if not pkl.is_file():
        msg = (
            f"surrogate artifact not found: {pkl}\n"
            "train first, e.g.:\n"
            "  PYTHONPATH=src uv run python -m microclimate.apps.train_surrogate "
            "--corpus /path/to/merged_corpus --out /path/to/artifact_dir\n"
            "then pass that same --out directory to run_case as --surrogate-artifact."
        )
        raise FileNotFoundError(msg)
    if not meta_path.is_file():
        msg = f"missing {meta_path} next to surrogate_params.pkl"
        raise FileNotFoundError(msg)
    with pkl.open("rb") as f:
        raw = pickle.load(f)  # noqa: S301
    params = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float32), raw)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return params, meta


def predict_field(
    params: object,
    problem_cfg: ProblemConfig,
    grid_cfg: GridConfig,
    activation: str,
    scales: dict[str, float],
) -> TemperatureField:
    """Flat params list matching ``microclimate.surrogate.model.Params``."""
    x, z = make_grid(problem_cfg, grid_cfg)
    x2, z2 = np.meshgrid(x, z, indexing="ij")
    xf = x2.ravel().astype(np.float32)
    zf = z2.ravel().astype(np.float32)
    xs = float(scales["x_scale"])
    zs = float(scales["z_scale"])
    us = float(scales["u_scale"])
    ts = float(scales["t_scale"])
    t0 = float(scales["t0"])
    ur = np.float32(problem_cfg.U_ref)
    tf = np.float32(problem_cfg.T_facade_hot)
    tr = np.float32(problem_cfg.T_ref)
    inp = np.stack(
        [
            xf / xs,
            zf / zs,
            np.full_like(xf, ur - 2.0) / us,
            np.full_like(xf, tf - t0) / ts,
            np.full_like(xf, tr - t0) / ts,
        ],
        axis=1,
    )
    pred = forward_batch(params, jnp.asarray(inp), activation)  # type: ignore[arg-type]
    t_out = np.asarray(pred, dtype=np.float64).reshape(x2.shape)
    solid = building_mask(x, z, problem_cfg)
    t_out = np.where(solid, np.nan, t_out)
    return TemperatureField(
        T=t_out,
        x_grid=x,
        z_grid=z,
        method_name="surrogate_mlp",
        runtime_seconds=0.0,
        config=problem_cfg,
    )
