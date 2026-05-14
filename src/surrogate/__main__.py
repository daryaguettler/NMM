"""entry: `PYTHONPATH=src uv run python -m surrogate --help`."""

from __future__ import annotations

import argparse
from pathlib import Path

from surrogate.corpus import load_particle_corpus_dir
from surrogate.training import train
from surrogate.types import load_config
from surrogate.validation import (
    validate_coupled,
    validate_inverse_design,
    validate_pressure_solver,
)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m surrogate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="run training from TrainingConfig json")
    t.add_argument("config", type=Path)

    v = sub.add_parser("validate", help="run validation triplet on one npz + corpus dir")
    v.add_argument("corpus_dir", type=Path)
    v.add_argument("npz", type=Path)

    args = ap.parse_args()
    if args.cmd == "train":
        cfg = load_config(args.config)
        train(cfg)
        return
    man = load_particle_corpus_dir(args.corpus_dir)
    topo, phy = man.topology, man.physics
    from surrogate.types import SurrogateConfig

    sur = SurrogateConfig()
    validate_pressure_solver(args.npz, topo, phy, sur=sur)
    validate_coupled(args.npz, topo, phy, sur)
    validate_inverse_design(args.npz, args.corpus_dir, topo, phy, sur, opt_steps=15)
    print("validate: ok")


if __name__ == "__main__":
    main()
