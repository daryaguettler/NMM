"""pydantic contracts for trajectories, corpus metadata, and linkage conventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# linkage order matches notebooks/final_project/visualizations.ipynb:
# per floor: window_front, window_back, crack_front, crack_back, doorway
N_ZONES = 6
N_LINKAGES = 15
N_WINDOWS = 6

SignConventionDoc = (
    "positive mass flow is from endpoint_a toward endpoint_b per linkage list order"
)


class TrajectoryArrays(BaseModel):
    """in-memory numpy bundle; validates shapes on construction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t: Any = Field(description="seconds (T,)")
    T_zones: Any = Field(description="degC (T, 6)")
    T_wall: Any = Field(description="degC (T, 6)")
    T_mass: Any = Field(description="degC (T, 6)")
    flows: Any = Field(description="kg/s signed (T, 15)")
    T_out: Any = Field(description="degC (T,)")
    wind_speed: Any = Field(description="m/s (T,)")
    wind_dir: Any = Field(description="degrees meteorological from (T,)")
    Q_sol: Any = Field(description="W per zone (T, 6)")
    Q_int: Any = Field(description="W per zone (T, 6)")
    openings: Any = Field(description="window factors (T, 6)")

    @field_validator(
        "t",
        "T_zones",
        "T_wall",
        "T_mass",
        "flows",
        "T_out",
        "wind_speed",
        "wind_dir",
        "Q_sol",
        "Q_int",
        "openings",
        mode="before",
    )
    @classmethod
    def _to_float_array(cls, v: Any) -> np.ndarray:
        return np.asarray(v, dtype=np.float64)

    @model_validator(mode="after")
    def _shapes(self) -> TrajectoryArrays:
        t = int(self.t.shape[0])
        if self.T_zones.shape != (t, N_ZONES):
            raise ValueError("T_zones must be (T, 6)")
        if self.flows.shape != (t, N_LINKAGES):
            raise ValueError("flows must be (T, 15)")
        if self.openings.shape != (t, N_WINDOWS):
            raise ValueError("openings must be (T, 6)")
        for name in ("T_wall", "T_mass", "Q_sol", "Q_int"):
            if getattr(self, name).shape != (t, N_ZONES):
                raise ValueError(f"{name} must be (T, 6)")
        for name in ("T_out", "wind_speed", "wind_dir"):
            if getattr(self, name).shape != (t,):
                raise ValueError(f"{name} must be (T,)")
        return self

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.t, dtype=np.float64),
            "T_zones": np.asarray(self.T_zones, dtype=np.float64),
            "T_wall": np.asarray(self.T_wall, dtype=np.float64),
            "T_mass": np.asarray(self.T_mass, dtype=np.float64),
            "flows": np.asarray(self.flows, dtype=np.float64),
            "T_out": np.asarray(self.T_out, dtype=np.float64),
            "wind_speed": np.asarray(self.wind_speed, dtype=np.float64),
            "wind_dir": np.asarray(self.wind_dir, dtype=np.float64),
            "Q_sol": np.asarray(self.Q_sol, dtype=np.float64),
            "Q_int": np.asarray(self.Q_int, dtype=np.float64),
            "openings": np.asarray(self.openings, dtype=np.float64),
        }

    @classmethod
    def from_npz_dict(cls, d: dict[str, np.ndarray]) -> TrajectoryArrays:
        payload = {k: np.asarray(v) for k, v in d.items() if k in cls.model_fields}
        return cls.model_validate(payload)


OpeningPolicy = Literal[
    "always_closed",
    "always_open",
    "daytime",
    "night_flush",
    "random",
    "threshold",
    "optimized",
    "synthetic_seed",
]


class RunManifestItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: int
    scenario_hash: str
    weather_window: str
    opening_policy: OpeningPolicy | str
    n_timesteps: int
    shard_id: int | None = None
    seed: int | None = None


class CorpusManifest(BaseModel):
    runs: list[RunManifestItem]
    sign_convention: str = SignConventionDoc


class CorpusConfigFile(BaseModel):
    """written next to manifest; echoes physics + version."""

    model_config = ConfigDict(extra="allow")

    generator: str = "particle_sim_v0"
    physics: dict[str, Any] = Field(default_factory=dict)
    geometry_note: str = "see particle_sim.config.spec"
    linkage_order_note: str = SignConventionDoc


def trajectory_from_npz_path(path: Path | str) -> TrajectoryArrays:
    p = Path(path)
    data = {k: v for k, v in np.load(p, allow_pickle=False).items()}
    return TrajectoryArrays.from_npz_dict(data)


def trajectory_to_npz(path: Path | str, traj: TrajectoryArrays) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **traj.to_npz_dict())


def manifest_to_json(path: Path | str, manifest: CorpusManifest) -> None:
    Path(path).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


def manifest_from_json(path: Path | str) -> CorpusManifest:
    return CorpusManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def corpus_config_to_json(path: Path | str, cfg: CorpusConfigFile) -> None:
    Path(path).write_text(
        json.dumps(cfg.model_dump(), indent=2),
        encoding="utf-8",
    )
