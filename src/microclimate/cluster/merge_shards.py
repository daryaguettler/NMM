"""merge shard_* microclimate corpora (same layout as particle_sim shards)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from microclimate.io.manifest import (
    MicroCorpusManifest,
    MicroRunRecord,
    read_manifest,
    write_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root: Path = args.shards_root
    out: Path = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    combined: list[MicroRunRecord] = []
    sign = "microclimate_v0"
    for shard in sorted(root.glob("shard_*")):
        mp = shard / "manifest.json"
        if not mp.exists():
            continue
        manifest = read_manifest(mp)
        if manifest.sign_convention:
            sign = manifest.sign_convention
        for item in manifest.runs:
            src = shard / "runs" / f"run_{item.run_id:06d}.npz"
            dst = out / "runs" / f"run_{item.run_id:06d}.npz"
            shutil.copy2(src, dst)
            meta = shard / "runs" / f"run_{item.run_id:06d}.meta.json"
            if meta.exists():
                shutil.copy2(meta, out / "runs" / meta.name)
            combined.append(item)
        cfg = shard / "config.json"
        if cfg.exists():
            shutil.copy2(cfg, out / "config.json")
    write_manifest(out / "manifest.json", MicroCorpusManifest(runs=combined, sign_convention=sign))


if __name__ == "__main__":
    main()
