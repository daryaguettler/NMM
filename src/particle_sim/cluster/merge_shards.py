"""merge shard_* directories into a single corpus with unique run ids."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from particle_sim.io.schema import CorpusManifest, manifest_from_json, manifest_to_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root: Path = args.shards_root
    out: Path = args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    all_runs: list = []
    sign: str = ""
    for shard in sorted(root.glob("shard_*")):
        mp = shard / "manifest.json"
        if not mp.exists():
            continue
        m = manifest_from_json(mp)
        if not sign:
            sign = m.sign_convention
        for item in m.runs:
            src = shard / "runs" / f"run_{item.run_id:06d}.npz"
            dst = out / "runs" / f"run_{item.run_id:06d}.npz"
            shutil.copy2(src, dst)
            meta = shard / "runs" / f"run_{item.run_id:06d}.meta.json"
            if meta.exists():
                shutil.copy2(meta, out / "runs" / meta.name)
            all_runs.append(item)
        cfg = shard / "config.json"
        if cfg.exists():
            shutil.copy2(cfg, out / "config.json")
    manifest_to_json(
        out / "manifest.json",
        CorpusManifest(runs=all_runs, sign_convention=sign or "see readme"),
    )


if __name__ == "__main__":
    main()
