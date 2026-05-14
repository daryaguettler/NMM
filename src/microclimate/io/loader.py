"""load helpers for microclimate artifacts."""

from __future__ import annotations

from pathlib import Path

from microclimate.io.writer import load_temperature_npz
from microclimate.types import TemperatureField


def load_run_temperature(path: Path) -> TemperatureField:
    return load_temperature_npz(path)
