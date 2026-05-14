"""jax mlp surrogate for microclimate t(x,z)."""

from microclimate.surrogate.data import (
    list_corpus_runs,
    load_all_runs,
    load_run_npz,
    run_to_flat_arrays,
)
from microclimate.surrogate.loss import combined_loss, mse_loss, physics_residual_loss
from microclimate.surrogate.model import forward, forward_batch, init_mlp, pack_inputs
from microclimate.surrogate.predict import load_surrogate_bundle, predict_field
from microclimate.surrogate.train import save_artifact, train_surrogate

__all__ = [
    "combined_loss",
    "forward",
    "forward_batch",
    "init_mlp",
    "list_corpus_runs",
    "load_all_runs",
    "load_run_npz",
    "load_surrogate_bundle",
    "mse_loss",
    "pack_inputs",
    "physics_residual_loss",
    "predict_field",
    "run_to_flat_arrays",
    "save_artifact",
    "train_surrogate",
]
