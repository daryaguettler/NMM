"""run analytic + pde [+ particles] [+ surrogate] and write figures/artifacts.

from repo root: ``PYTHONPATH=src uv run python -m microclimate.apps.run_case``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from microclimate.config import (
    GridConfig,
    ParticleConfig,
    PDESolverConfig,
    ProblemConfig,
)
from microclimate.grid import building_mask, metrics_rmse_max
from microclimate.io.writer import write_meta_json, write_temperature_npz
from microclimate.methods.analytic import solve_analytic_field
from microclimate.methods.particles import solve_particle_field
from microclimate.methods.pde import solve_pde_field
from microclimate.surrogate.predict import load_surrogate_bundle, predict_field
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
    p.add_argument("--particles", action="store_true", help="run lagrangian ensemble (after pde)")
    p.add_argument("--n-particles", type=int, default=5000)
    p.add_argument("--particle-spinup", type=int, default=800)
    p.add_argument("--particle-avg", type=int, default=400)
    p.add_argument(
        "--surrogate-artifact",
        type=Path,
        default=None,
        help="train_surrogate --out directory (contains surrogate_params.pkl + surrogate_meta.json)",
    )
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

    fields: list = [fa, fp]
    titles: list[str] = ["analytic", "pde"]

    pcfg = ParticleConfig(
        n_particles=int(args.n_particles),
        n_steps_spinup=int(args.particle_spinup),
        n_steps_average=int(args.particle_avg),
        seed=42,
    )
    f_part = None
    if args.particles:
        f_part = solve_particle_field(cfg, grid, pcfg, wind_field=fp)
        fields.append(f_part)
        titles.append("particles")

    f_surr = None
    if args.surrogate_artifact is not None:
        params, meta = load_surrogate_bundle(Path(args.surrogate_artifact))
        t0 = time.perf_counter()
        f_surr = predict_field(
            params,
            cfg,
            grid,
            str(meta["activation"]),
            meta["scales"],
        )
        rt = time.perf_counter() - t0
        f_surr = f_surr.model_copy(update={"runtime_seconds": float(rt)})
        fields.append(f_surr)
        titles.append("surrogate")

    for fn, path in [
        (fa, out / "analytic.npz"),
        (fp, out / "pde.npz"),
    ]:
        write_temperature_npz(path, fn)
    if f_part is not None:
        write_temperature_npz(out / "particles.npz", f_part)
    if f_surr is not None:
        write_temperature_npz(out / "surrogate.npz", f_surr)

    fluid = ~building_mask(fa.x_grid, fa.z_grid, cfg)
    metrics: dict = {
        "rmse_analytic_vs_pde_degC": metrics_rmse_max(fa.T, fp.T, fluid)[0],
        "max_abs_analytic_vs_pde_degC": metrics_rmse_max(fa.T, fp.T, fluid)[1],
        "analytic_runtime_s": fa.runtime_seconds,
        "pde_runtime_s": fp.runtime_seconds,
    }
    if f_part is not None:
        m0, m1 = metrics_rmse_max(f_part.T, fp.T, fluid)
        metrics["rmse_particles_vs_pde_degC"] = m0
        metrics["max_abs_particles_vs_pde_degC"] = m1
        metrics["particles_runtime_s"] = f_part.runtime_seconds
    if f_surr is not None:
        m0, m1 = metrics_rmse_max(f_surr.T, fp.T, fluid)
        metrics["rmse_surrogate_vs_pde_degC"] = m0
        metrics["max_abs_surrogate_vs_pde_degC"] = m1
        metrics["surrogate_runtime_s"] = f_surr.runtime_seconds

    write_meta_json(out / "metrics.json", metrics)

    fig = plot_method_grid(fields, titles, cfg, out_path=out / "compare_T.png")
    close_fig(fig)

    z_a, col_a = facade_profile(fa, cfg)
    _z_p, col_p = facade_profile(fp, cfg)
    prof: dict[str, object] = {"analytic": col_a, "pde": col_p}
    if f_part is not None:
        _z_pt, col_pt = facade_profile(f_part, cfg)
        prof["particles"] = col_pt
    if f_surr is not None:
        _z_s, col_s = facade_profile(f_surr, cfg)
        prof["surrogate"] = col_s
    fig2 = plot_vertical_profiles(z_a, prof, cfg, out_path=out / "facade_profile.png")
    close_fig(fig2)

    print(json.dumps({"out": str(out), **{k: metrics[k] for k in sorted(metrics)}}, indent=2))


if __name__ == "__main__":
    main()
