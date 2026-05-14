"""json manifest for microclimate corpus shards (phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MicroRunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: int
    note: str = ""


class MicroCorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs: list[MicroRunRecord]
    sign_convention: str = "microclimate_v0"


def write_manifest(path: Path, manifest: MicroCorpusManifest) -> None:
    path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")


def read_manifest(path: Path) -> MicroCorpusManifest:
    return MicroCorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
