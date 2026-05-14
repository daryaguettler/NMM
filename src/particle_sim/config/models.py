"""validated config models for the particle simulator and corpus builder."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field


class GeometrySpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    long_m: float = Field(12.0, gt=0)
    wide_m: float = Field(7.0, gt=0)
    ceiling_m: float = Field(3.0, gt=0)
    floor_gap_m: float = Field(3.0, ge=0)
    n_floors: int = Field(3, ge=1, le=6)

    @computed_field
    @property
    def half_depth_m(self) -> float:
        return self.wide_m / 2.0


class NumericsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dt_sim: float = Field(10.0, gt=0, le=300.0)
    dt_output: float = Field(600.0, gt=0)
    n_particles_per_zone: int = Field(300, ge=10, le=5000)
    substeps_per_output: int | None = Field(
        None,
        description="if None, int(dt_output/dt_sim)",
    )
    k_rep: float = Field(2.0, ge=0.0)
    r_cut: float = Field(1.0, gt=0.0)
    damping_coeff: float = Field(0.2, ge=0.0)
    v_max: float = Field(5.0, gt=0.0)
    buoyancy_strength: float = Field(0.15, ge=0.0)
    forcing_layer_depth: float = Field(0.5, gt=0)

    # thermal
    thermal_layer_depth: float = Field(0.3, gt=0)
    rho_air: float = Field(1.2, gt=0)
    cp_air: float = Field(1005.0, gt=0)

    passage_window: float = Field(1.0, ge=0.0, le=1.0)
    passage_doorway: float = Field(1.0, ge=0.0, le=1.0)
    passage_crack: float = Field(0.3, ge=0.0, le=1.0)

    def resolved_substeps(self) -> int:
        if self.substeps_per_output is not None:
            return int(self.substeps_per_output)
        assert self.dt_output % self.dt_sim < 1e-9 or (
            abs(self.dt_output / self.dt_sim - round(self.dt_output / self.dt_sim)) < 1e-6
        ), "dt_output must be multiple of dt_sim"
        return int(round(self.dt_output / self.dt_sim))


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    h_int: float = Field(7.7, gt=0)
    c_wall_j_per_k: float = Field(4.8e6, gt=0)
    c_mass_j_per_k: float = Field(4.8e6, gt=0)
    ua_out_w_per_k: float = Field(16.0, gt=0)
    h_a_mass_w_per_k: float = Field(120.0, ge=0)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_runs: int = Field(500, ge=1)
    duration_hours: float = Field(168.0, gt=0)
    global_seed: int = Field(0)
    source_tag: str = "v0_particle"


class SimConfig(BaseModel):
    """full stack used by simulate and corpus apps."""

    model_config = ConfigDict(extra="forbid")

    geometry: GeometrySpec = Field(default_factory=GeometrySpec)
    numerics: NumericsConfig = Field(default_factory=NumericsConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
