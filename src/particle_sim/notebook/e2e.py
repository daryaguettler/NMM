"""end-to-end helpers for notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from particle_sim.config.defaults import default_sim_config
from particle_sim.core.simulator import simulate_trajectory
from particle_sim.io import loader
from particle_sim.io.schema import (
    CorpusManifest,
    RunManifestItem,
    TrajectoryArrays,
    manifest_to_json,
)
from particle_sim.io.writer import scenario_hash, write_corpus_config, write_run_npz
from particle_sim.scenarios.sampler import resample_for_sim, sample_scenario


def run_demo_scenario(
    *,
    seed: int = 0,
    duration_hours: float = 24.0,
    particles: int = 80,
) -> TrajectoryArrays:
    cfg = default_sim_config()
    cfg.corpus.duration_hours = duration_hours
    cfg.numerics.n_particles_per_zone = particles
    rng = np.random.default_rng(seed)
    sc = sample_scenario(cfg, rng)
    T_out, ws, wd, qs, qi, op = resample_for_sim(sc, cfg)
    return simulate_trajectory(cfg, T_out, ws, wd, qs, qi, op, seed=seed)


def load_demo_run(
    corpus_dir: str | Path,
    run_id: int = 0,
) -> TrajectoryArrays:
    return loader.load_corpus_run(
        Path(corpus_dir) / "runs" / f"run_{run_id:06d}.npz",
    )


def write_smoke_corpus(
    out_dir: str | Path = Path("corpus/v0_particle_smoke"),
    *,
    n_runs: int = 1,
    particles: int = 40,
    seed: int = 42,
    duration_hours: float = 6.0,
) -> Path:
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    cfg = default_sim_config()
    cfg.corpus.duration_hours = float(duration_hours)
    cfg.numerics.n_particles_per_zone = int(particles)
    runs: list[RunManifestItem] = []
    for rid in range(n_runs):
        rng = np.random.default_rng(seed + rid)
        sc = sample_scenario(cfg, rng)
        T_out, ws, wd, qs, qi, op = resample_for_sim(sc, cfg)
        h = scenario_hash({"policy": sc.policy, "seed": seed + rid})
        traj = simulate_trajectory(cfg, T_out, ws, wd, qs, qi, op, seed=seed + rid)
        item = RunManifestItem(
            run_id=rid,
            scenario_hash=h,
            weather_window=f"synthetic_seed={seed + rid}",
            opening_policy=str(sc.policy),
            n_timesteps=int(traj.t.shape[0]),
            seed=int(seed + rid),
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
    write_corpus_config(out / "config.json", json.loads(cfg.model_dump_json()))
    return out
