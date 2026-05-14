"""pydantic configs shared across microclimate methods."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProblemConfig(BaseModel):
    """shared physical and domain parameters."""

    model_config = ConfigDict(frozen=True)

    x_min: float = 0.0
    x_max: float = 60.0
    z_min: float = 0.0
    z_max: float = 30.0

    bldg_x_min: float = 25.0
    bldg_x_max: float = 35.0
    bldg_z_min: float = 0.0
    bldg_z_max: float = 10.0

    U_ref: float = 5.0
    z_ref: float = 10.0
    z_0: float = 0.5
    T_ref: float = 25.0
    T_facade_hot: float = 50.0
    T_ground: float = 25.0

    g: float = 9.81
    beta: float = 1.0 / 298.15
    nu: float = 1.5e-5
    alpha: float = 2.1e-5


class GridConfig(BaseModel):
    """cell counts; dx, dz derived from ProblemConfig bounds."""

    model_config = ConfigDict(frozen=True)

    nx: int = Field(default=200, ge=4)
    nz: int = Field(default=100, ge=4)


class PDESolverConfig(BaseModel):
    """pseudo-steady iteration for streamfunction-vorticity + temperature."""

    model_config = ConfigDict(frozen=True)

    max_outer_iters: int = 8000
    tol_T: float = 1e-4
    tol_psi: float = 1e-5
    # under-relaxation for T, omega, psi updates
    relax_T: float = 0.35
    relax_omega: float = 0.25
    relax_psi: float = 0.6
    # poisson (psi) jacobi sweeps per outer iteration
    poisson_sweeps: int = 40
    # optional: solve advection-diffusion for T only with frozen log-law u (debug)
    freeze_mean_wind_only: bool = False
    # scale molecular diffusivities for coarse-grid stability (still coupled)
    nu_scale: float = 60.0
    alpha_scale: float = 40.0
    max_inner_T_adv: float = 2.5


class SurrogateConfig(BaseModel):
    """neural surrogate (phase 2)."""

    model_config = ConfigDict(frozen=True)

    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    activation: Literal["gelu", "relu", "tanh"] = "gelu"
    learning_rate: float = 1e-3
    n_iterations: int = 5000
    physics_weight: float = 0.0
    n_collocation_per_batch: int = 256
    seed: int = 0
