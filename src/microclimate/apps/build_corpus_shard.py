"""slurm shard worker: each task writes a disjoint run_id range of pde solves."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from microclimate.config import GridConfig, PDESolverConfig, ProblemConfig
from microclimate.io.manifest import MicroCorpusManifest, MicroRunRecord, write_manifest
from microclimate.io.writer import write_meta_json, write_temperature_npz
from microclimate.methods.pde import solve_pde_field

_U_GRID = (3.0, 5.0, 7.0)
_F_GRID = (35.0, 50.0, 65.0)
_T_GRID = (20.0, 25.0, 30.0)
_CASES: list[tuple[float, float, float]] = list(itertools.product(_U_GRID, _F_GRID, _T_GRID))


def _case_params(run_id: int) -> tuple[float, float, float]:
    return _CASES[int(run_id) % len(_CASES)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-start", type=int, required=True)
    p.add_argument("--run-count", type=int, required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--pde-iters", type=int, default=4000)
    args = p.parse_args()
    shard_dir = args.out / f"shard_{args.shard_id:04d}"
    (shard_dir / "runs").mkdir(parents=True, exist_ok=True)

    grid = GridConfig(nx=int(args.nx), nz=int(args.nz))
    solver = PDESolverConfig(max_outer_iters=int(args.pde_iters))
    runs: list[MicroRunRecord] = []

    for k in range(args.run_count):
        rid = int(args.run_start) + k
        u_ref, t_fac, t_ref = _case_params(rid)
        prob = ProblemConfig(
            U_ref=float(u_ref),
            T_facade_hot=float(t_fac),
            T_ref=float(t_ref),
            T_ground=float(t_ref),
        )
        field = solve_pde_field(prob, grid, solver)
        npz_path = shard_dir / "runs" / f"run_{rid:06d}.npz"
        write_temperature_npz(npz_path, field)
        meta = {
            "run_id": rid,
            "shard_id": int(args.shard_id),
            "global_seed": int(args.global_seed),
            "U_ref": u_ref,
            "T_facade_hot": t_fac,
            "T_ref": t_ref,
            "pde_runtime_s": field.runtime_seconds,
        }
        write_meta_json(shard_dir / "runs" / f"run_{rid:06d}.meta.json", meta)
        runs.append(MicroRunRecord(run_id=rid, note=json.dumps(meta)))

    write_manifest(
        shard_dir / "manifest.json",
        MicroCorpusManifest(runs=runs, sign_convention="microclimate_v0_pde"),
    )
    combo_manifest = {
        "nx": grid.nx,
        "nz": grid.nz,
        "pde_outer_iters": solver.max_outer_iters,
        "n_parameter_combos": len(_CASES),
    }
    (shard_dir / "config.json").write_text(json.dumps(combo_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
