"""run default analytic + pde case and write figures/artifacts.

from repo root: ``PYTHONPATH=src uv run python -m microclimate.apps.run_case``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microclimate.config import GridConfig, PDESolverConfig, ProblemConfig
from microclimate.grid import building_mask, metrics_rmse_max
from microclimate.io.writer import write_meta_json, write_temperature_npz
from microclimate.methods.analytic import solve_analytic_field
from microclimate.methods.pde import solve_pde_field
from microclimate.viz.plots import (
    close_fig,
    facade_profile,
    plot_method_grid,
    plot_vertical_profiles,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/microclimate/default_run"))
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--nz", type=int, default=60)
    p.add_argument("--pde-iters", type=int, default=4000)
    p.add_argument("--freeze-wind", action="store_true", help="advection-diffusion only (debug)")
    args = p.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = ProblemConfig()
    grid = GridConfig(nx=args.nx, nz=args.nz)
    sol = PDESolverConfig(
        max_outer_iters=args.pde_iters,
        freeze_mean_wind_only=bool(args.freeze_wind),
    )

    fa = solve_analytic_field(cfg, grid)
    fp = solve_pde_field(cfg, grid, sol)

    write_temperature_npz(out / "analytic.npz", fa)
    write_temperature_npz(out / "pde.npz", fp)

    fluid = ~building_mask(fa.x_grid, fa.z_grid, cfg)
    rmse, mx = metrics_rmse_max(fa.T, fp.T, fluid)
    write_meta_json(
        out / "metrics.json",
        {
            "rmse_vs_pde_degC": rmse,
            "max_abs_vs_pde_degC": mx,
            "analytic_runtime_s": fa.runtime_seconds,
            "pde_runtime_s": fp.runtime_seconds,
        },
    )

    fig = plot_method_grid(
        [fa, fp],
        ["analytic", "pde"],
        cfg,
        out_path=out / "compare_T.png",
    )
    close_fig(fig)

    z_a, col_a = facade_profile(fa, cfg)
    _z_p, col_p = facade_profile(fp, cfg)
    fig2 = plot_vertical_profiles(
        z_a,
        {"analytic": col_a, "pde": col_p},
        cfg,
        out_path=out / "facade_profile.png",
    )
    close_fig(fig2)

    print(json.dumps({"out": str(out), "rmse_degC": rmse, "max_abs_degC": mx}, indent=2))


if __name__ == "__main__":
    main()
