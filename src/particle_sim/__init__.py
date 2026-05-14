"""particle-cfd v0 synthetic corpus generator."""

from particle_sim.config.defaults import default_sim_config, sim_config_from_spec_path
from particle_sim.config.models import SimConfig
from particle_sim.core.simulator import simulate_trajectory
from particle_sim.io.loader import (
    corpus_root,
    forcings_dict,
    load_corpus_run,
    state_dict,
)
from particle_sim.io.schema import TrajectoryArrays

__all__ = [
    "SimConfig",
    "TrajectoryArrays",
    "corpus_root",
    "default_sim_config",
    "forcings_dict",
    "load_corpus_run",
    "sim_config_from_spec_path",
    "simulate_trajectory",
    "state_dict",
]
