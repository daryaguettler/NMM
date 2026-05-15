"""slurm shard worker: each task writes a disjoint run_id range of indoor PDE solves.

Parameter grid (18 cases total):
    T_facade_hot  : [35, 45, 55] °C
    T_outdoor     : [20, 25, 30] °C
    window_open   : [0.0, 1.0]

from repo root (local test):
    PYTHONPATH=src uv run python -m indoor.apps.build_corpus_shard \
        --out /tmp/indoor_shards --run-start 0 --run-count 3 --shard-id 0
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from indoor.config import IndoorGridConfig, IndoorPDESolverConfig, IndoorProblemConfig
from indoor.methods.pde import solve_pde_field

_T_HOT_GRID    = (35.0, 45.0, 55.0)
_T_OUT_GRID    = (20.0, 25.0, 30.0)
_WIN_OPEN_GRID = (0.0, 1.0)

_CASES: list[tuple[float, float, float]] = list(
    itertools.product(_T_HOT_GRID, _T_OUT_GRID, _WIN_OPEN_GRID)
)


def _case_params(run_id: int) -> tuple[float, float, float]:
    return _CASES[int(run_id) % len(_CASES)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-start", type=int, required=True)
    p.add_argument("--run-count", type=int, required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=240)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--pde-iters", type=int, default=15_000)
    args = p.parse_args()

    shard_dir = args.out / f"shard_{args.shard_id:04d}"
    (shard_dir / "runs").mkdir(parents=True, exist_ok=True)

    grid   = IndoorGridConfig(nx=int(args.nx), nz=int(args.nz))
    solver = IndoorPDESolverConfig(max_outer_iters=int(args.pde_iters))

    manifest_rows: list[dict[str, object]] = []

    for k in range(args.run_count):
        rid = int(args.run_start) + k
        t_hot, t_out, win = _case_params(rid)
        prob = IndoorProblemConfig(
            T_facade_hot=float(t_hot),
            T_outdoor=float(t_out),
            window_open=float(win),
        )
        field = solve_pde_field(prob, grid, solver)

        npz_path = shard_dir / "runs" / f"run_{rid:06d}.npz"
        arrays: dict[str, object] = {"T": field.T, "x_grid": field.x_grid, "z_grid": field.z_grid}
        if field.u is not None:
            arrays["u"] = field.u
        if field.w is not None:
            arrays["w"] = field.w
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "run_id": rid,
            "shard_id": int(args.shard_id),
            "global_seed": int(args.global_seed),
            "T_facade_hot": t_hot,
            "T_outdoor": t_out,
            "window_open": win,
            "pde_runtime_s": field.runtime_seconds,
        }
        (shard_dir / "runs" / f"run_{rid:06d}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        manifest_rows.append(meta)
        print(
            f"shard={args.shard_id} run={rid} "
            f"T_hot={t_hot} T_out={t_out} win={win} "
            f"t={field.runtime_seconds:.1f}s"
        )

    combo_config = {
        "nx": grid.nx,
        "nz": grid.nz,
        "pde_outer_iters": solver.max_outer_iters,
        "n_parameter_combos": len(_CASES),
        "T_hot_grid": list(_T_HOT_GRID),
        "T_outdoor_grid": list(_T_OUT_GRID),
        "window_open_grid": list(_WIN_OPEN_GRID),
    }
    (shard_dir / "config.json").write_text(json.dumps(combo_config, indent=2), encoding="utf-8")
    (shard_dir / "manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
