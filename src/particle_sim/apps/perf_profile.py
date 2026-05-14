"""optional timing helper for batch sizing on engaging."""

from __future__ import annotations

import argparse
import time

import numpy as np

from particle_sim.config.defaults import default_sim_config
from particle_sim.core.simulator import simulate_trajectory
from particle_sim.scenarios.sampler import resample_for_sim, sample_scenario


def main() -> None:
    """cli entry."""
    p = argparse.ArgumentParser()
    p.add_argument("--particles", type=int, default=300)
    p.add_argument("--hours", type=float, default=4.0)
    args = p.parse_args()
    cfg = default_sim_config()
    cfg.corpus.duration_hours = float(args.hours)
    cfg.numerics.n_particles_per_zone = int(args.particles)
    rng = np.random.default_rng(0)
    sc = sample_scenario(cfg, rng)
    T_out, ws, wd, qs, qi, op = resample_for_sim(sc, cfg)
    t0 = time.perf_counter()
    simulate_trajectory(cfg, T_out, ws, wd, qs, qi, op, seed=0)
    print("elapsed_s", round(time.perf_counter() - t0, 3))


if __name__ == "__main__":
    main()
