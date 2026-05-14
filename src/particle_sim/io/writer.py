"""write npz runs, manifest, config."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from particle_sim.io.schema import (
    CorpusManifest,
    RunManifestItem,
    TrajectoryArrays,
    manifest_to_json,
    trajectory_to_npz,
)

DEFAULT_SIGN = (
    "positive mass flow is from endpoint_a toward endpoint_b per linkage list order"
)


def write_corpus_config(path: Path | str, data: dict[str, Any]) -> None:
    p = Path(path)
    payload = dict(data)
    payload.setdefault("generator", "particle_sim_v0")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_manifest(path: Path | str, item: RunManifestItem, *, reset: bool = False) -> None:
    p = Path(path)
    if reset or not p.exists():
        manifest = CorpusManifest(runs=[item], sign_convention=DEFAULT_SIGN)
    else:
        m = CorpusManifest.model_validate_json(p.read_text(encoding="utf-8"))
        m.runs.append(item)
        manifest = m
    manifest_to_json(p, manifest)


def scenario_hash(blob: dict[str, Any]) -> str:
    s = json.dumps(blob, sort_keys=True, default=str).encode()
    return hashlib.sha256(s).hexdigest()[:20]


def write_run_npz(
    out_dir: Path | str,
    run_id: int,
    traj: TrajectoryArrays,
    meta: RunManifestItem,
) -> Path:
    root = Path(out_dir)
    run_dir = root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / f"run_{run_id:06d}.npz"
    meta_path = run_dir / f"run_{run_id:06d}.meta.json"
    trajectory_to_npz(fp, traj)
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return fp

