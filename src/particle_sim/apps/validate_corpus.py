"""sanity-check a corpus directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from particle_sim.core.aggregation import max_abs_residual_over_time
from particle_sim.io.schema import manifest_from_json, trajectory_from_npz_path


def main() -> None:
    """cli entry."""
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--max-residual", type=float, default=0.2)
    args = p.parse_args()
    root = args.corpus
    man = manifest_from_json(root / "manifest.json")
    worst = 0.0
    for item in man.runs[: min(5, len(man.runs))]:
        path = root / "runs" / f"run_{item.run_id:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        traj = trajectory_from_npz_path(path)
        w = max_abs_residual_over_time(traj.flows)
        worst = max(worst, w)
        if traj.T_zones.shape[1] != 6:
            raise ValueError("expected 6 zones")
    print("max kirchhoff residual (sampled)", worst)
    if worst > args.max_residual:
        raise SystemExit(f"kirchhoff residual {worst} > {args.max_residual}")


if __name__ == "__main__":
    main()
