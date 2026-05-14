"""save / load npz runs + sidecar meta."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microclimate.config import ProblemConfig
from microclimate.types import TemperatureField


def write_temperature_npz(path: Path, field: TemperatureField) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "T": np.asarray(field.T),
        "x_grid": np.asarray(field.x_grid),
        "z_grid": np.asarray(field.z_grid),
        "method_name": np.asarray(field.method_name),
        "runtime_seconds": np.asarray(field.runtime_seconds, dtype=np.float64),
        "problem_config_json": np.asarray(field.config.model_dump_json()),
    }
    if field.u is not None:
        payload["u"] = np.asarray(field.u)
    if field.w is not None:
        payload["w"] = np.asarray(field.w)
    np.savez_compressed(path, **payload)


def write_meta_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_temperature_npz(path: Path) -> TemperatureField:
    data = np.load(path, allow_pickle=True)
    cfg_raw = data["problem_config_json"]
    cfg_js = cfg_raw.item() if hasattr(cfg_raw, "item") else cfg_raw
    cfg = ProblemConfig.model_validate_json(str(cfg_js))
    u_arr = data["u"] if "u" in data.files else None
    w_arr = data["w"] if "w" in data.files else None
    mn = data["method_name"]
    method_name = str(mn.item() if hasattr(mn, "item") else mn)
    rt = data["runtime_seconds"]
    runtime_seconds = float(rt.item() if hasattr(rt, "item") else rt)
    return TemperatureField(
        T=np.asarray(data["T"]),
        x_grid=np.asarray(data["x_grid"]),
        z_grid=np.asarray(data["z_grid"]),
        method_name=method_name,
        runtime_seconds=runtime_seconds,
        config=cfg,
        u=None if u_arr is None else np.asarray(u_arr),
        w=None if w_arr is None else np.asarray(w_arr),
    )
