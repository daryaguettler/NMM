"""pydantic configs for the indoor microclimate project."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IndoorProblemConfig(BaseModel):
    """geometry, boundary conditions, and physics for one indoor scenario."""

    model_config = ConfigDict(frozen=True)

    # ── domain ────────────────────────────────────────────────────────────────
    x_min: float = 0.0
    x_max: float = 12.0   # front-to-back depth (m)
    z_min: float = 0.0
    z_max: float = 3.0    # floor-to-ceiling (m)

    # ── interior partitions ───────────────────────────────────────────────────
    # x-coordinates of full-height partition walls
    partition_x: tuple[float, ...] = (4.0, 8.0)
    # doorway opening in each partition: z ∈ [doorway_z_lo, doorway_z_hi]
    doorway_z_lo: float = 0.0
    doorway_z_hi: float = 2.0   # 2 m tall doorway; header from 2–3 m

    # ── window openings ───────────────────────────────────────────────────────
    # south window: left wall (x = x_min), inflow
    win_south_z_lo: float = 1.0
    win_south_z_hi: float = 2.0
    # north window: right wall (x = x_max), outflow
    win_north_z_lo: float = 1.0
    win_north_z_hi: float = 2.0
    # 0 = fully closed, 1 = fully open
    window_open: float = 1.0
    U_window: float = 0.5  # inflow speed when open (m/s)

    # ── temperatures (°C) ─────────────────────────────────────────────────────
    T_ref: float = 25.0          # reference / buoyancy base
    T_facade_hot: float = 45.0   # sun-heated south facade (left wall)
    T_facade_cool: float = 25.0  # shaded north facade (right wall)
    T_floor: float = 25.0
    T_ceiling: float = 27.0      # slightly warm (absorbed radiation)
    T_outdoor: float = 30.0      # temperature of incoming window air

    # ── physics ───────────────────────────────────────────────────────────────
    g: float = 9.81
    beta: float = 1.0 / 298.15
    nu: float = 1.5e-5
    alpha: float = 2.1e-5


class IndoorGridConfig(BaseModel):
    """cell counts for the indoor domain; dx/dz derived from config bounds."""

    model_config = ConfigDict(frozen=True)

    nx: int = Field(default=240, ge=4)
    nz: int = Field(default=100, ge=4)


class IndoorPDESolverConfig(BaseModel):
    """pseudo-steady iteration for closed-room streamfunction-vorticity."""

    model_config = ConfigDict(frozen=True)

    max_outer_iters: int = 15_000   # more than outdoor: buoyancy-dominated
    tol_T: float = 1e-5             # tighter: indoor gradients are subtler
    tol_psi: float = 1e-5
    relax_T: float = 0.25
    relax_omega: float = 0.20
    relax_psi: float = 0.55
    poisson_sweeps: int = 50
    nu_scale: float = 80.0          # artificial viscosity for stability
    alpha_scale: float = 50.0
    max_inner_T_adv: float = 2.0


class IndoorParticleConfig(BaseModel):
    """lagrangian ensemble for closed indoor room."""

    model_config = ConfigDict(frozen=True)

    n_particles: int = Field(default=20_000, ge=16)
    dt: float = Field(default=0.05, gt=0.0)
    n_steps_spinup: int = Field(default=5_000, ge=0)
    n_steps_average: int = Field(default=4_000, ge=1)
    n_steps_record: int = Field(default=800, ge=0)
    record_stride: int = Field(default=10, ge=1)
    turbulent_intensity: float = Field(default=0.12, ge=0.0)
    T_L: float = Field(default=2.0, gt=0.0)
    velocity_relax: float = Field(default=0.10, ge=0.0, le=1.0)
    # thermal relaxation at each surface (dimensionless fraction per step)
    relax_hot_wall: float = 0.20
    relax_cool_wall: float = 0.15
    relax_floor: float = 0.25
    relax_ceiling: float = 0.12
    relax_partition: float = 0.00   # adiabatic partitions
    seed: int = 0
    use_pde_wind: bool = True


class IndoorSurrogateConfig(BaseModel):
    """mlp surrogate trained on indoor pde corpus."""

    model_config = ConfigDict(frozen=True)

    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    activation: Literal["gelu", "relu", "tanh"] = "gelu"
    learning_rate: float = 1e-3
    n_iterations: int = 5_000
    physics_weight: float = 0.0
    n_collocation_per_batch: int = 256
    seed: int = 0


# ── notebook-friendly lighter configs ─────────────────────────────────────────

NOTEBOOK_PROBLEM = IndoorProblemConfig()

NOTEBOOK_GRID = IndoorGridConfig(nx=120, nz=50)

NOTEBOOK_PDE = IndoorPDESolverConfig(
    max_outer_iters=3_000,
    tol_T=1e-4,
    tol_psi=1e-4,
)

NOTEBOOK_PARTICLES = IndoorParticleConfig(
    n_particles=6_000,
    n_steps_spinup=2_000,
    n_steps_average=1_500,
    n_steps_record=300,
)
