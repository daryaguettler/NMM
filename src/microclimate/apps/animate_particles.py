"""particle motion movie (gif/mp4) over pde wind + optional t background.

2d physics (x–z boussinesq + particles); ``--three-d`` only changes the *camera*
(extruded scene along an illustrative across-wind axis `y`).

from repo root:
  PYTHONPATH=src uv run python -m microclimate.apps.animate_particles --out outputs/microclimate/parts.gif

mp4 requires ffmpeg installed; gif uses pillow only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.patches as mpatches

from microclimate.config import GridConfig, ParticleConfig, PDESolverConfig, ProblemConfig
from microclimate.methods.particles import sample_particle_trajectories
from microclimate.methods.pde import solve_pde_field


def _box_face_verts(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> list[np.ndarray]:
    return [
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
        np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),
        np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),
    ]


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
    p.add_argument(
        "--three-d",
        action="store_true",
        help="extruded 3d view (physics still 2d x–z)",
    )
    p.add_argument(
        "--scene-depth",
        type=float,
        default=8.0,
        help="half-width (m) along illustrative y for extrusion",
    )
    p.add_argument(
        "--viz-building2",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
        default=None,
        help="optional second block drawn in 3d only — not in pde/particles physics",
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

    rng_py = np.random.default_rng(123)
    py_jitter = rng_py.uniform(
        -float(args.scene_depth), float(args.scene_depth), size=int(args.n_trace)
    ).astype(np.float64)
    depth = float(args.scene_depth)
    y0, y1 = -depth, depth
    vis_note = ""

    if args.viz_building2 is not None:
        bx0, bx1, bz0, bz1 = [float(v) for v in args.viz_building2]
        vis_note = " | 2nd block = viz only (not in solver)"

    if not args.three_d:
        qskip = int(args.quiver_skip)
        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

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
            ax.set_title(
                f"particles  t={float(t_snap[frame]):.2f}s  (pde wind, dt={pcfg.dt})"
            )
            return []

        anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    else:
        fig = plt.figure(figsize=(9.0, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        norm_t = Normalize(vmin=vmin, vmax=vmax)
        qskip = int(args.quiver_skip)
        rstride = max(1, int(args.nx) // 50)
        cstride = max(1, int(args.nz) // 50)

        def _draw_building(ax3: Any, xa: float, xb: float, za: float, zb: float, *, alpha: float) -> None:
            faces = _box_face_verts(xa, xb, y0, y1, za, zb)
            poly = Poly3DCollection(
                faces,
                facecolors=(0.25, 0.25, 0.28, alpha),
                edgecolor="k",
                linewidths=0.4,
            )
            ax3.add_collection3d(poly)

        def update3(frame: int) -> list:
            ax.cla()
            Yplane = np.zeros_like(X, dtype=np.float64)
            if not args.no_t_field:
                Tpaint = np.where(np.isfinite(Tm), Tm, vmin)
                rgba = cm.inferno(norm_t(Tpaint))
                ax.plot_surface(
                    X,
                    Yplane,
                    Z,
                    rstride=rstride,
                    cstride=cstride,
                    facecolors=rgba,
                    shade=False,
                    linewidth=0,
                    antialiased=True,
                )
            _draw_building(
                ax,
                cfg.bldg_x_min,
                cfg.bldg_x_max,
                cfg.bldg_z_min,
                cfg.bldg_z_max,
                alpha=0.95,
            )
            if args.viz_building2 is not None:
                _draw_building(ax, bx0, bx1, bz0, bz1, alpha=0.45)

            if qskip > 0 and fp.u is not None and fp.w is not None:
                uu = np.asarray(fp.u)
                ww = np.asarray(fp.w)
                Yq = np.zeros_like(X)
                ax.quiver(
                    X[::qskip, ::qskip],
                    Yq[::qskip, ::qskip],
                    Z[::qskip, ::qskip],
                    uu[::qskip, ::qskip],
                    np.zeros_like(uu[::qskip, ::qskip]),
                    ww[::qskip, ::qskip],
                    length=1.2,
                    normalize=True,
                    color="w",
                    alpha=0.55,
                )

            ax.scatter(
                px[frame],
                py_jitter,
                pz[frame],
                c=pT[frame],
                cmap="coolwarm",
                s=14.0,
                alpha=0.82,
                vmin=float(cfg.T_ref),
                vmax=float(cfg.T_facade_hot),
                depthshade=True,
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m) — illustrative across-wind")
            ax.set_zlabel("z (m)")
            ax.set_title(
                f"3d view of 2d simulation  t={float(t_snap[frame]):.2f}s{vis_note}"
            )
            lx = cfg.x_max - cfg.x_min
            lz = cfg.z_max - cfg.z_min
            ly = 2.0 * depth
            ax.set_box_aspect((lx, ly, lz))
            ax.view_init(elev=22, azim=-58)
            ax.set_xlim(cfg.x_min, cfg.x_max)
            ax.set_ylim(y0, y1)
            ax.set_zlim(cfg.z_min, cfg.z_max)
            return []

        anim = animation.FuncAnimation(fig, update3, frames=n_frames, blit=False)

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
