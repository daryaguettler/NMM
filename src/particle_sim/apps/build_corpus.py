"""generate corpus/v0_particle locally."""

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
    p.add_argument("--out", type=Path, default=Path("corpus/v0_particle"))
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--particles", type=int, default=300)
    p.add_argument("--duration-hours", type=float, default=None)
    args = p.parse_args()
    out: Path = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    cfg: SimConfig = default_sim_config()
    cfg.numerics.n_particles_per_zone = int(args.particles)
    if args.duration_hours is not None:
        cfg.corpus.duration_hours = float(args.duration_hours)
    runs: list[RunManifestItem] = []
    for rid in range(args.n_runs):
        rng = np.random.default_rng(args.seed + rid)
        sc = sample_scenario(cfg, rng)
        T_out, ws, wd, qs, qi, op = resample_for_sim(sc, cfg)
        meta_blob = {
            "policy": sc.policy,
            "duration": cfg.corpus.duration_hours,
            "seed": int(args.seed + rid),
        }
        h = scenario_hash(meta_blob)
        traj = simulate_trajectory(
            cfg,
            T_out,
            ws,
            wd,
            qs,
            qi,
            op,
            seed=int(args.seed + rid),
        )
        item = RunManifestItem(
            run_id=rid,
            scenario_hash=h,
            weather_window=f"synthetic_seed={args.seed + rid}",
            opening_policy=str(sc.policy),
            n_timesteps=int(traj.t.shape[0]),
            seed=int(args.seed + rid),
        )
        runs.append(item)
        write_run_npz(out, rid, traj, item)

    manifest_to_json(
        out / "manifest.json",
        CorpusManifest(
            runs=runs,
            sign_convention=(
                "positive mass flow is from endpoint_a toward endpoint_b per linkage"
            ),
        ),
    )
    full_cfg = json.loads(cfg.model_dump_json())
    write_corpus_config(out / "config.json", full_cfg)


if __name__ == "__main__":
    main()
