"""load pde corpus npz runs for surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from microclimate.config import ProblemConfig


@dataclass(frozen=True)
class RunArrays:
    path: Path
    cfg: ProblemConfig
    x_grid: np.ndarray
    z_grid: np.ndarray
    T: np.ndarray
    u: np.ndarray | None
    w: np.ndarray | None
    fluid: np.ndarray


def _fluid_mask(x: np.ndarray, z: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    x2 = x[:, np.newaxis]
    z2 = z[np.newaxis, :]
    bldg = (
        (x2 >= cfg.bldg_x_min)
        & (x2 <= cfg.bldg_x_max)
        & (z2 >= cfg.bldg_z_min)
        & (z2 <= cfg.bldg_z_max)
    )
    return ~bldg


def load_run_npz(path: Path) -> RunArrays:
    data = np.load(path, allow_pickle=True)
    cfg_js = data["problem_config_json"]
    cfg_raw = cfg_js.item() if hasattr(cfg_js, "item") else cfg_js
    cfg = ProblemConfig.model_validate_json(str(cfg_raw))
    x = np.asarray(data["x_grid"], dtype=np.float64)
    z = np.asarray(data["z_grid"], dtype=np.float64)
    T = np.asarray(data["T"], dtype=np.float64)
    u = np.asarray(data["u"], dtype=np.float64) if "u" in data.files else None
    w = np.asarray(data["w"], dtype=np.float64) if "w" in data.files else None
    fluid = _fluid_mask(x, z, cfg)
    return RunArrays(path=path, cfg=cfg, x_grid=x, z_grid=z, T=T, u=u, w=w, fluid=fluid)


def list_corpus_runs(corpus_dir: Path) -> list[Path]:
    root = Path(corpus_dir)
    return sorted((root / "runs").glob("run_*.npz"))


def load_all_runs(corpus_dir: Path) -> list[RunArrays]:
    return [load_run_npz(p) for p in list_corpus_runs(corpus_dir)]


def run_to_flat_arrays(
    run: RunArrays, max_points: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    x2, z2 = np.meshgrid(run.x_grid, run.z_grid, indexing="ij")
    x_flat = x2.reshape(-1)
    z_flat = z2.reshape(-1)
    t_flat = run.T.reshape(-1)
    m_flat = run.fluid.reshape(-1)
    sel = m_flat & np.isfinite(t_flat)
    x_f = x_flat[sel]
    z_f = z_flat[sel]
    t_f = t_flat[sel]
    u_f = run.u.reshape(-1)[sel] if run.u is not None else None
    w_f = run.w.reshape(-1)[sel] if run.w is not None else None
    if max_points is not None and x_f.size > max_points:
        choice = rng.choice(x_f.size, size=max_points, replace=False)
        x_f = x_f[choice]
        z_f = z_f[choice]
        t_f = t_f[choice]
        if u_f is not None:
            u_f = u_f[choice]
        if w_f is not None:
            w_f = w_f[choice]
    weather = np.array(
        [run.cfg.U_ref, run.cfg.T_facade_hot, run.cfg.T_ref], dtype=np.float64
    )
    w_block = np.tile(weather, (x_f.size, 1))
    return x_f, z_f, w_block, t_f, u_f, w_f
