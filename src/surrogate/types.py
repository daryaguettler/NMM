"""pydantic boundaries: topology, physics, scenarios, trajectories, training."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

LinkageKind = Literal["window", "doorway", "crack"]
ZoneSide = Literal["front", "back"]
OpeningPolicyName = Literal[
    "always_closed",
    "always_open",
    "daytime",
    "night_flush",
    "threshold",
    "random",
]
CorpusSource = Literal["particle_sim", "energyplus_afn"]


class ZoneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    floor: int = Field(ge=1, le=3)
    side: ZoneSide
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )


class LinkageSpecT(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: LinkageKind
    a: str
    b: str
    x: float
    y: float
    width: float
    facade_azimuth: float | None = None


class Topology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    zones: list[ZoneSpec]
    linkages: list[LinkageSpecT]

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    @property
    def n_linkages(self) -> int:
        return len(self.linkages)

    def zone_index(self, name: str) -> int:
        for i, z in enumerate(self.zones):
            if z.name == name:
                return i
        raise KeyError(name)

    def linkage_index(self, name: str) -> int:
        for i, k in enumerate(self.linkages):
            if k.name == name:
                return i
        raise KeyError(name)

    @model_validator(mode="after")
    def _check_endpoints(self) -> Topology:
        zone_names = {z.name for z in self.zones}
        for lk in self.linkages:
            if lk.a not in zone_names:
                raise ValueError(f"linkage {lk.name} endpoint a={lk.a} not a zone")
            if lk.b not in zone_names and lk.b != "outdoor":
                raise ValueError(f"linkage {lk.name} endpoint b={lk.b} invalid")
        return self


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rho_air: float = 1.2
    cp_air: float = 1005.0
    C_air_per_zone: float = 57.9e3
    C_wall_per_zone: float = 4.8e6
    C_mass_per_zone: float = 4.8e6
    hA_wall: float = 185.0
    hA_mass: float = 246.0
    UA_outside_per_zone: float = 16.0
    C_window: float = 0.65
    n_window: float = 0.5
    C_doorway: float = 0.78
    n_doorway: float = 0.5
    C_crack: float = 0.001
    n_crack: float = 0.65
    Cp_amplitude: float = 0.6
    dt_output: float = 600.0
    dt_solver: float = 1e-3


class WeatherWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    n_hours: int = Field(default=168, ge=24)
    daily_mean_temp: float = 24.0
    diurnal_range: float = 8.0
    heatwave_peak_amp: float = 0.0
    heatwave_center_hour: float | None = None
    wind_mean_speed: float = 2.5
    wind_prevailing_dir: float = 225.0


class OpeningPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: OpeningPolicyName
    open_hour: float | None = None
    close_hour: float | None = None
    threshold_T: float | None = None
    threshold_dT_out: float | None = None
    switch_rate_hours: float | None = None
    seed: int = 0
    per_zone_independent: bool = False


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: int
    weather: WeatherWindow
    opening: OpeningPolicy
    physics: PhysicsConfig
    topology_name: str

    @property
    def scenario_hash(self) -> str:
        payload = self.model_dump_json(exclude={"scenario_id"})
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Forcings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    T_out: Any
    wind_speed: Any
    wind_dir: Any
    Q_sol: Any
    Q_int: Any
    openings: Any
    dt: float

    @model_validator(mode="after")
    def _axis(self) -> Forcings:
        t = int(np.asarray(self.T_out).shape[0])
        n_z = int(np.asarray(self.Q_sol).shape[1])
        nw = int(np.asarray(self.openings).shape[1])
        assert np.asarray(self.wind_speed).shape[0] == t
        assert np.asarray(self.wind_dir).shape[0] == t
        assert np.asarray(self.Q_int).shape == (t, n_z)
        assert np.asarray(self.openings).shape[0] == t
        _ = nw
        return self

    @property
    def n_steps(self) -> int:
        return int(np.asarray(self.T_out).shape[0])


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    T_air: Any
    T_wall: Any
    T_mass: Any


class Trajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    t: Any
    T_air: Any
    T_wall: Any
    T_mass: Any
    flows: Any
    T_out: Any
    wind_speed: Any
    wind_dir: Any
    openings: Any
    Q_sol: Any | None = None
    Q_int: Any | None = None
    particle_snapshots: Any | None = None

    @property
    def n_steps(self) -> int:
        return int(np.asarray(self.t).shape[0])

    def to_npz(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "t": np.asarray(self.t, dtype=np.float64),
            "T_zones": np.asarray(self.T_air, dtype=np.float64),
            "T_wall": np.asarray(self.T_wall, dtype=np.float64),
            "T_mass": np.asarray(self.T_mass, dtype=np.float64),
            "flows": np.asarray(self.flows, dtype=np.float64),
            "T_out": np.asarray(self.T_out, dtype=np.float64),
            "wind_speed": np.asarray(self.wind_speed, dtype=np.float64),
            "wind_dir": np.asarray(self.wind_dir, dtype=np.float64),
            "Q_sol": np.asarray(self.Q_sol, dtype=np.float64) if self.Q_sol is not None else np.zeros((self.n_steps, 6)),
            "Q_int": np.asarray(self.Q_int, dtype=np.float64) if self.Q_int is not None else np.zeros((self.n_steps, 6)),
            "openings": np.asarray(self.openings, dtype=np.float64),
        }
        np.savez_compressed(p, **payload)

    @classmethod
    def from_npz(cls, path: Path | str) -> Trajectory:
        d = dict(np.load(path, allow_pickle=False))
        return cls(
            t=d["t"],
            T_air=d["T_zones"],
            T_wall=d["T_wall"],
            T_mass=d["T_mass"],
            flows=d["flows"],
            T_out=d["T_out"],
            wind_speed=d["wind_speed"],
            wind_dir=d["wind_dir"],
            openings=d["openings"],
            Q_sol=d.get("Q_sol"),
            Q_int=d.get("Q_int"),
        )


class CorpusRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: int
    scenario_hash: str
    scenario: ScenarioSpec | None = None
    path: str
    n_timesteps: int
    source: CorpusSource = "particle_sim"
    weather_seed: int | None = None


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    topology: Topology
    physics: PhysicsConfig
    n_runs: int
    runs: list[CorpusRun]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def split(
        self,
        val_fraction: float,
        seed: int,
    ) -> tuple[CorpusManifest, CorpusManifest]:
        rng = np.random.default_rng(seed)
        seeds: dict[int, list[CorpusRun]] = {}
        for r in self.runs:
            ws = r.weather_seed
            if ws is None:
                ws = 0
            seeds.setdefault(ws, []).append(r)
        all_seeds = sorted(seeds.keys())
        n_val = int(len(all_seeds) * val_fraction)
        if len(all_seeds) > 1:
            if val_fraction <= 0.0:
                n_val = 0
            else:
                n_val = min(max(n_val, 1), len(all_seeds) - 1)
        else:
            n_val = 0
        rng.shuffle(all_seeds)
        val_seed_set = set(all_seeds[:n_val])
        tr, va = [], []
        for s, grp in seeds.items():
            if s in val_seed_set:
                va.extend(grp)
            else:
                tr.extend(grp)
        train = CorpusManifest(
            name=self.name + "_train",
            topology=self.topology,
            physics=self.physics,
            n_runs=len(tr),
            runs=sorted(tr, key=lambda x: x.run_id),
            created_at=self.created_at,
        )
        val = CorpusManifest(
            name=self.name + "_val",
            topology=self.topology,
            physics=self.physics,
            n_runs=len(va),
            runs=sorted(va, key=lambda x: x.run_id),
            created_at=self.created_at,
        )
        return train, val


class SurrogateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_pressure_solver: bool = True
    use_heat_balance: bool = True
    use_learned_residual: bool = False
    residual_hidden_sizes: tuple[int, ...] = (32, 32)
    residual_activation: Literal["relu", "gelu", "tanh"] = "gelu"
    newton_max_iter: int = 25
    newton_tol: float = 1e-6
    pressure_solver_implicit_diff: bool = True
    variant_name: str = "pure_physics"


class SurrogateParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    residual_mlp_params: dict[str, Any] | None = None
    residual_mlp_activation: Literal["relu", "gelu", "tanh"] = "gelu"


class LossWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 1.0
    flows: float = 0.1
    wall_temp: float = 0.0
    mass_temp: float = 0.0
    flow_window: float = 1.0
    flow_doorway: float = 1.0
    flow_crack: float = 0.5
    gradient_supervision: float = 0.0
    gradient_supervision_input: Literal["openings", "Q_sol"] = "openings"
    heatwave_weight_T_threshold: float = 28.0
    heatwave_weight_multiplier: float = 1.0
    normalize_flows_per_linkage: bool = True


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    schedule: Literal["constant", "cosine", "exponential"] = "constant"
    schedule_decay_steps: int = 1000


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    corpus_path: str
    val_fraction: float = 0.2
    val_split_seed: int = 0
    batch_size: int = 8
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    n_epochs: int = 100
    grad_clip: float | None = 1.0
    weights: LossWeights = Field(default_factory=LossWeights)
    surrogate: SurrogateConfig = Field(default_factory=SurrogateConfig)
    seed: int = 0
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    checkpoint_every_n_epochs: int = 10
    output_dir: str = "runs/surrogate_default"


class TrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_params: SurrogateParams
    train_losses: Any
    val_losses: Any
    val_metrics_by_epoch: list[dict[str, float]]
    config: TrainingConfig
    runtime_seconds: float
    final_val_rmse_temperature: float
    final_val_rmse_flow: float


class PressureSolverValidation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flows_true: Any
    flows_pred: Any
    rmse_per_linkage: Any
    r2_per_linkage: Any
    bias_per_linkage: Any


class CoupledValidation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    T_air_true: Any
    T_air_pred: Any
    flows_true: Any
    flows_pred: Any
    rmse_T_per_zone: Any
    rmse_T_per_run: Any
    hours_above_threshold_true: Any
    hours_above_threshold_pred: Any
    hours_above_threshold_error: Any


class InverseDesignValidation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_openings: Any
    optimized_openings: Any
    surrogate_loss_trajectory: Any
    surrogate_final_hours_above: float
    simulator_final_hours_above: float
    discrepancy: float


class SimulatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    particles_per_zone: int = 40
    duration_hours: float = 168.0
    deterministic: bool = True
    base_seed: int = 0


def save_config(config: TrainingConfig, path: Path | str) -> None:
    Path(path).write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_config(path: Path | str) -> TrainingConfig:
    return TrainingConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
