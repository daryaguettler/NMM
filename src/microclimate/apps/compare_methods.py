"""compare saved npz fields or recompute; print metrics and optional plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microclimate.config import GridConfig, PDESolverConfig, ProblemConfig
from microclimate.grid import building_mask, metrics_rmse_max
from microclimate.io.writer import load_temperature_npz
from microclimate.methods.analytic import solve_analytic_field
from microclimate.methods.pde import solve_pde_field
from microclimate.viz.plots import close_fig, facade_profile, plot_vertical_profiles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analytic-npz", type=Path, default=None)
    p.add_argument("--pde-npz", type=Path, default=None)
    p.add_argument("--plot", type=Path, default=None)
    p.add_argument("--nx", type=int, default=80)
    p.add_argument("--nz", type=int, default=40)
    p.add_argument("--pde-iters", type=int, default=2500)
    args = p.parse_args()

    if args.analytic_npz and args.pde_npz:
        a = load_temperature_npz(args.analytic_npz)
        b = load_temperature_npz(args.pde_npz)
        cfg = a.config
    else:
        cfg = ProblemConfig()
        grid = GridConfig(nx=args.nx, nz=args.nz)
        sol = PDESolverConfig(max_outer_iters=args.pde_iters)
        a = solve_analytic_field(cfg, grid)
        b = solve_pde_field(cfg, grid, sol)

    fluid = ~building_mask(a.x_grid, a.z_grid, cfg)
    rmse, mx = metrics_rmse_max(a.T, b.T, fluid)
    payload = {"rmse_degC": rmse, "max_abs_degC": mx}
    print(json.dumps(payload, indent=2))

    if args.plot is not None:
        z_, ca = facade_profile(a, cfg)
        _z2, cb = facade_profile(b, cfg)
        fig = plot_vertical_profiles(z_, {"analytic": ca, "pde": cb}, cfg, out_path=args.plot)
        close_fig(fig)


if __name__ == "__main__":
    main()
