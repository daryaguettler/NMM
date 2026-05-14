"""train mlp on merged microclimate pde corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from microclimate.config import SurrogateConfig
from microclimate.surrogate.train import save_artifact, train_surrogate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True, help="merged corpus dir with runs/")
    p.add_argument("--out", type=Path, required=True, help="artifact directory")
    p.add_argument("--physics-weight", type=float, default=0.0, help="0=variant a, >0 variant b")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of runs for validation (ignored when only one run exists)",
    )
    p.add_argument("--max-points", type=int, default=8000)
    args = p.parse_args()

    cfg = SurrogateConfig(
        learning_rate=float(args.lr),
        n_iterations=int(args.iterations),
        physics_weight=float(args.physics_weight),
        seed=int(args.seed),
    )
    bundle = train_surrogate(
        args.corpus,
        cfg,
        val_fraction=float(args.val_fraction),
        max_points_per_run=int(args.max_points) if args.max_points > 0 else None,
    )
    save_artifact(Path(args.out), bundle, cfg)
    print(
        f"wrote {args.out}  train_runs={bundle['train_runs']} "
        f"val_runs={bundle['val_runs']} wall_s={bundle['wall_seconds']:.2f}"
    )


if __name__ == "__main__":
    main()
