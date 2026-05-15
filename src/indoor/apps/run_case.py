"""run all four indoor methods and write figures + artifacts.

from repo root:
    PYTHONPATH=src uv run python -m indoor.apps.run_case
    PYTHONPATH=src uv run python -m indoor.apps.run_case --particles --pde-iters 8000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from indoor.config import (
    IndoorGridConfig,
    IndoorParticleConfig,
    IndoorPDESolverConfig,
    IndoorProblemConfig,
)
from indoor.grid import metrics_rmse_max, partition_mask
from indoor.methods.analytic import solve_analytic_field
from indoor.methods.particles import solve_particle_field
from indoor.methods.pde import solve_pde_field
from indoor.types import IndoorField


def _write_npz(path: Path, field: IndoorField) -> None:
    arrays: dict[str, object] = {
        "T": field.T,
        "x_grid": field.x_grid,
        "z_grid": field.z_grid,
    }
    if field.u is not None:
        arrays["u"] = field.u
    if field.w is not None:
        arrays["w"] = field.w
    np.savez_compressed(path, **arrays)


def main() -> None:
    p = argparse.ArgumentParser(description="Run indoor four-method comparison.")
    p.add_argument("--out", type=Path, default=Path("outputs/indoor/default_run"))
    p.add_argument("--nx", type=int, default=240)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--pde-iters", type=int, default=15_000)
    p.add_argument("--T-hot", type=float, default=45.0, dest="T_hot")
    p.add_argument("--T-outdoor", type=float, default=30.0, dest="T_outdoor")
    p.add_argument("--window-open", type=float, default=1.0)
    p.add_argument("--particles", action="store_true")
    p.add_argument("--n-particles", type=int, default=20_000)
    p.add_argument("--particle-spinup", type=int, default=5_000)
    p.add_argument("--particle-avg", type=int, default=4_000)
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = IndoorProblemConfig(
        T_facade_hot=float(args.T_hot),
        T_outdoor=float(args.T_outdoor),
        window_open=float(args.window_open),
    )
    grid    = IndoorGridConfig(nx=args.nx, nz=args.nz)
    solver  = IndoorPDESolverConfig(max_outer_iters=args.pde_iters)

    # Method 1: analytic (microseconds)
    fa = solve_analytic_field(cfg, grid)
    _write_npz(out / "analytic.npz", fa)

    # Method 2: PDE (reference truth for indoor)
    fp = solve_pde_field(cfg, grid, solver)
    _write_npz(out / "pde.npz", fp)

    fields: list[IndoorField] = [fa, fp]
    titles: list[str] = ["analytic", "pde"]

    # Method 3: particles (optional — expected to fail indoors)
    f_part = None
    if args.particles:
        pcfg = IndoorParticleConfig(
            n_particles=int(args.n_particles),
            n_steps_spinup=int(args.particle_spinup),
            n_steps_average=int(args.particle_avg),
        )
        f_part = solve_particle_field(cfg, grid, pcfg, wind_field=fp)
        _write_npz(out / "particles.npz", f_part)
        fields.append(f_part)
        titles.append("particles")

    # metrics vs PDE reference
    solid = partition_mask(fp.x_grid, fp.z_grid, cfg)
    fluid = ~solid & ~np.isnan(fp.T)

    def _safe_metrics(a: IndoorField) -> tuple[float, float]:
        mask = fluid & ~np.isnan(a.T)
        return metrics_rmse_max(
            np.nan_to_num(a.T), np.nan_to_num(fp.T), mask
        )

    rmse_a, mx_a = _safe_metrics(fa)
    metrics: dict[str, object] = {
        "T_hot": cfg.T_facade_hot,
        "T_outdoor": cfg.T_outdoor,
        "window_open": cfg.window_open,
        "rmse_analytic_vs_pde_degC": rmse_a,
        "max_abs_analytic_vs_pde_degC": mx_a,
        "analytic_runtime_s": fa.runtime_seconds,
        "pde_runtime_s": fp.runtime_seconds,
    }
    if f_part is not None:
        rm, mx = _safe_metrics(f_part)
        metrics["rmse_particles_vs_pde_degC"] = rm
        metrics["max_abs_particles_vs_pde_degC"] = mx
        metrics["particles_runtime_s"] = f_part.runtime_seconds

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in sorted(metrics)}, indent=2))


if __name__ == "__main__":
    main()
