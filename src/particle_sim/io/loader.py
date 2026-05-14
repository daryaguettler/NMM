"""load trajectory npz for notebooks and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from particle_sim.io.schema import TrajectoryArrays, trajectory_from_npz_path

CorpusKind = Literal["v0_particle", "v0_energyplus"]


def corpus_root(kind: CorpusKind, base: Path | str | None = None) -> Path:
    root = Path(base) if base is not None else Path("corpus")
    return root / kind


def load_corpus_run(
    run_path: Path | str,
    *,
    base: Path | str | None = None,
    kind: CorpusKind = "v0_particle",
) -> TrajectoryArrays:
    """Load a single run; path can be relative to repo or absolute."""
    p = Path(run_path)
    if not p.exists() and base is not None:
        p = corpus_root(kind, base) / "runs" / run_path
    if not p.exists():
        p = corpus_root(kind) / "runs" / Path(run_path).name
    return trajectory_from_npz_path(p)


def to_numpy_backend(traj: TrajectoryArrays) -> dict[str, np.ndarray]:
    return traj.to_npz_dict()


def forcings_dict(traj: TrajectoryArrays) -> dict[str, Any]:
    """notebook-friendly bundle matching figure data patterns."""
    d = traj.to_npz_dict()
    return {
        "t": d["t"],
        "T_out": d["T_out"],
        "wind_speed": d["wind_speed"],
        "wind_dir": d["wind_dir"],
        "Q_sol": d["Q_sol"],
        "Q_int": d["Q_int"],
        "openings": d["openings"],
        "flows": d["flows"],
    }


def state_dict(traj: TrajectoryArrays) -> dict[str, Any]:
    return {
        "T_zones": traj.T_zones,
        "T_wall": traj.T_wall,
        "T_mass": traj.T_mass,
    }
