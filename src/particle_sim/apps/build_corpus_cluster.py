"""shard runner for slurm arrays: each task writes runs in a disjoint id range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from particle_sim.config.defaults import default_sim_config
from particle_sim.config.models import SimConfig
from particle_sim.core.simulator import simulate_trajectory
from particle_sim.io.schema import CorpusManifest, RunManifestItem, manifest_to_json
from particle_sim.io.writer import scenario_hash, write_corpus_config, write_run_npz
from particle_sim.scenarios.sampler import resample_for_sim, sample_scenario


def main() -> None:
    """cli entry."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-start", type=int, required=True)
    p.add_argument("--run-count", type=int, required=True)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--particles", type=int, default=300)
    p.add_argument("--duration-hours", type=float, default=None)
    args = p.parse_args()
    out: Path = args.out
    shard_dir = out / f"shard_{args.shard_id:04d}"
    (shard_dir / "runs").mkdir(parents=True, exist_ok=True)
    cfg: SimConfig = default_sim_config()
    cfg.numerics.n_particles_per_zone = int(args.particles)
    if args.duration_hours is not None:
        cfg.corpus.duration_hours = float(args.duration_hours)
    runs: list[RunManifestItem] = []
    for k in range(args.run_count):
        rid = args.run_start + k
        seed = args.global_seed + rid * 10007 + args.shard_id
        rng = np.random.default_rng(seed)
        sc = sample_scenario(cfg, rng)
        T_out, ws, wd, qs, qi, op = resample_for_sim(sc, cfg)
        h = scenario_hash({"policy": sc.policy, "seed": seed, "rid": rid})
        traj = simulate_trajectory(cfg, T_out, ws, wd, qs, qi, op, seed=seed)
        item = RunManifestItem(
            run_id=rid,
            scenario_hash=h,
            weather_window=f"synthetic_seed={seed}",
            opening_policy=str(sc.policy),
            n_timesteps=int(traj.t.shape[0]),
            seed=int(seed),
            shard_id=int(args.shard_id),
        )
        runs.append(item)
        write_run_npz(shard_dir, rid, traj, item)
    manifest_to_json(
        shard_dir / "manifest.json",
        CorpusManifest(
            runs=runs,
            sign_convention=(
                "positive mass flow is from endpoint_a toward endpoint_b per linkage"
            ),
        ),
    )
    write_corpus_config(shard_dir / "config.json", json.loads(cfg.model_dump_json()))


if __name__ == "__main__":
    main()
