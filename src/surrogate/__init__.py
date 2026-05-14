"""surrogate package public surface."""

from surrogate.corpus import (
    compute_flow_scale_rms,
    initial_state_from_trajectory,
    load_particle_corpus_dir,
    load_run_trajectory,
    load_sim_config,
    trajectory_to_forcings,
)
from surrogate.loss import (
    finite_difference_gradient,
    gradient_supervision_loss,
    trajectory_supervision_loss,
)
from surrogate.surrogate import apply_residual_mlp, init_residual_mlp, simulate
from surrogate.topology_builder import build_default_topology
from surrogate.training import train
from surrogate.types import (
    CorpusManifest,
    Forcings,
    LossWeights,
    OptimizerConfig,
    SurrogateConfig,
    SurrogateParams,
    TrainingConfig,
    TrainingResult,
    load_config,
    save_config,
)
from surrogate.validation import (
    validate_coupled,
    validate_inverse_design,
    validate_pressure_solver,
)

__all__ = [
    "CorpusManifest",
    "Forcings",
    "LossWeights",
    "OptimizerConfig",
    "SurrogateConfig",
    "SurrogateParams",
    "TrainingConfig",
    "TrainingResult",
    "apply_residual_mlp",
    "build_default_topology",
    "compute_flow_scale_rms",
    "finite_difference_gradient",
    "gradient_supervision_loss",
    "init_residual_mlp",
    "initial_state_from_trajectory",
    "load_config",
    "load_particle_corpus_dir",
    "load_run_trajectory",
    "load_sim_config",
    "save_config",
    "simulate",
    "train",
    "trajectory_supervision_loss",
    "trajectory_to_forcings",
    "validate_coupled",
    "validate_inverse_design",
    "validate_pressure_solver",
]
