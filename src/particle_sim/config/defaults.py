"""load default sim config from pydantic models + optional spec json."""

from __future__ import annotations

import json
from pathlib import Path

from particle_sim.config.models import (
    CorpusConfig,
    GeometrySpec,
    NumericsConfig,
    PhysicsConfig,
    SimConfig,
)


def default_sim_config() -> SimConfig:
    return SimConfig()


def sim_config_from_spec_path(path: Path | str) -> SimConfig:
    p = Path(path)
    blob = json.loads(p.read_text(encoding="utf-8"))
    return SimConfig(
        geometry=GeometrySpec(**blob["geometry"]),
        numerics=NumericsConfig(**blob["numerics"]),
        physics=PhysicsConfig(**blob["physics"]),
        corpus=CorpusConfig(),
    )
