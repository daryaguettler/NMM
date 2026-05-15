"""particle motion movie (gif/mp4) over pde wind + optional t background.

from repo root:
  PYTHONPATH=src uv run python -m microclimate.apps.animate_particles --out outputs/microclimate/parts.gif

mp4 requires ffmpeg installed; gif uses pillow only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from microclimate.config import GridConfig, ParticleConfig, PDESolverConfig, ProblemConfig
from microclimate.methods.particles import sample_particle_trajectories
from microclimate.methods.pde import solve_pde_field


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--nz", type=int, default=60)
    p.add_argument("--pde-iters", type=int, default=4000)
    p.add_argument("--n-trace", type=int, default=500)
    p.add_argument("--spinup", type=int, default=400)
    p.add_argument("--record-steps", type=int, default=1000)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--no-t-field", action="store_true")
    p.add_argument(
        "--quiver-skip",
        type=int,
        default=0,
        help="if >0, draw wind quiver with this index stride on the pde grid",
    )
    args = p.parse_args()

    cfg = ProblemConfig()
    grid = GridConfig(nx=int(args.nx), nz=int(args.nz))
    sol = PDESolverConfig(max_outer_iters=int(args.pde_iters))
    fp = solve_pde_field(cfg, grid, sol)

    pcfg = ParticleConfig(
        n_particles=int(args.n_trace),
        n_steps_spinup=1,
        n_steps_average=1,
        dt=0.05,
        seed=42,
    )
    traj = sample_particle_trajectories(
        cfg,
        grid,
        pcfg,
        fp,
        n_trace=int(args.n_trace),
        spinup_steps=int(args.spinup),
        record_steps=int(args.record_steps),
        record_stride=int(args.stride),
    )
    px = traj["px"]
    pz = traj["pz"]
    pT = traj["pT"]
    t_snap = traj["t"]
    n_frames = int(px.shape[0])

    x = np.asarray(fp.x_grid)
    z = np.asarray(fp.z_grid)
    X, Z = np.meshgrid(x, z, indexing="ij")
    Tm = np.asarray(fp.T)
    finite = np.isfinite(Tm)
    vmin = float(np.nanmin(Tm)) if np.any(finite) else cfg.T_ref - 5.0
    vmax = float(np.nanmax(Tm)) if np.any(finite) else cfg.T_facade_hot + 5.0

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    qskip = int(args.quiver_skip)

    def update(frame: int) -> list:
        ax.clear()
        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.z_min, cfg.z_max)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal")
        if not args.no_t_field:
            Tplot = np.ma.masked_where(~np.isfinite(Tm), Tm)
            ax.pcolormesh(
                X,
                Z,
                Tplot,
                shading="auto",
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                alpha=0.88,
            )
        ax.add_patch(
            mpatches.Rectangle(
                (cfg.bldg_x_min, cfg.bldg_z_min),
                cfg.bldg_x_max - cfg.bldg_x_min,
                cfg.bldg_z_max - cfg.bldg_z_min,
                fill=True,
                facecolor="0.33",
                edgecolor="k",
                linewidth=0.8,
                zorder=5,
            )
        )
        if qskip > 0 and fp.u is not None and fp.w is not None:
            uu = np.asarray(fp.u)
            ww = np.asarray(fp.w)
            ax.quiver(
                X[::qskip, ::qskip],
                Z[::qskip, ::qskip],
                uu[::qskip, ::qskip],
                ww[::qskip, ::qskip],
                scale=55.0,
                width=0.0025,
                color="w",
                alpha=0.65,
                zorder=4,
            )
        ax.scatter(
            px[frame],
            pz[frame],
            c=pT[frame],
            cmap="coolwarm",
            s=10.0,
            alpha=0.78,
            vmin=float(cfg.T_ref),
            vmax=float(cfg.T_facade_hot),
            edgecolors="none",
            zorder=6,
        )
        ax.set_title(f"particles  t={float(t_snap[frame]):.2f}s  (pde wind, dt={pcfg.dt})")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    suf = out.suffix.lower()
    if suf == ".gif":
        anim.save(out, writer="pillow", fps=int(args.fps), dpi=int(args.dpi))
    elif suf == ".mp4":
        anim.save(out, writer="ffmpeg", fps=int(args.fps), dpi=int(args.dpi))
    else:
        raise SystemExit("--out must end with .gif or .mp4")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
