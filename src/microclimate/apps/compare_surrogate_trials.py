"""compare trained surrogate artifacts: training curves + optional field error vs pde.

from repo root:
  PYTHONPATH=src uv run python -m microclimate.apps.compare_surrogate_trials \\
    --artifact path/to/run_a --artifact path/to/run_b \\
    --out outputs/microclimate/trial_compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from microclimate.config import GridConfig, PDESolverConfig, ProblemConfig
from microclimate.grid import building_mask, metrics_rmse_max
from microclimate.methods.pde import solve_pde_field
from microclimate.surrogate.predict import load_surrogate_bundle, predict_field
from microclimate.viz.plots import (
    close_fig,
    plot_surrogate_field_errors,
    plot_surrogate_training_curves,
)


def _steps_train_val_from_meta(meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """recover logged curves from surrogate_meta.json (full history or legacy tail)."""
    sc = meta.get("surrogate_config") or {}
    n_it = int(sc.get("n_iterations") or 0)
    full = meta.get("history")
    if isinstance(full, dict) and full.get("train_loss"):
        tr = np.asarray(full["train_loss"], dtype=np.float64)
        va = np.asarray(full["val_loss"], dtype=np.float64)
        st = full.get("step")
        if st is not None and len(st) == len(tr):
            steps = np.asarray(st, dtype=np.float64)
        elif n_it > 0 and tr.size:
            steps = np.linspace(0, n_it - 1, tr.size)
        else:
            steps = np.arange(tr.size, dtype=np.float64)
        return steps, tr, va
    tail = meta.get("history_tail") or {}
    tr = np.asarray(tail.get("train_loss") or (), dtype=np.float64)
    va = np.asarray(tail.get("val_loss") or (), dtype=np.float64)
    if tr.size == 0:
        return np.array([]), np.array([]), np.array([])
    if n_it > 0:
        steps = np.linspace(0, n_it - 1, tr.size)
    else:
        steps = np.arange(tr.size, dtype=np.float64)
    return steps, tr, va


def _trial_training_row(artifact: Path, label: str, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _s, tr, va = _steps_train_val_from_meta(meta)
    row: dict = {
        "label": label,
        "artifact": str(artifact.resolve()),
        "wall_seconds": float(meta.get("wall_seconds") or 0.0),
        "surrogate_config": meta.get("surrogate_config"),
        "final_train_loss": float(tr[-1]) if tr.size else None,
        "final_val_loss": float(va[-1]) if va.size else None,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifact",
        type=Path,
        action="append",
        dest="artifacts",
        required=True,
        help="train_surrogate --out directory (repeat per trial)",
    )
    p.add_argument(
        "--label",
        type=str,
        action="append",
        dest="labels",
        default=None,
        help="legend label (same order as --artifact; default: directory name)",
    )
    p.add_argument("--out", type=Path, required=True, help="report directory")
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--nz", type=int, default=60)
    p.add_argument("--pde-iters", type=int, default=4000)
    p.add_argument(
        "--skip-field-eval",
        action="store_true",
        help="only training-curve plots + json (no pde reference solve)",
    )
    args = p.parse_args()

    artifacts = [Path(a).expanduser().resolve() for a in args.artifacts]
    if args.labels is not None and len(args.labels) != len(artifacts):
        p.error("count of --label must match count of --artifact")
    labels = list(args.labels) if args.labels else [a.name for a in artifacts]

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    training_rows: list[dict] = []
    for art, lab in zip(artifacts, labels, strict=True):
        mp = art / "surrogate_meta.json"
        if not mp.is_file():
            raise FileNotFoundError(f"missing {mp}")
        training_rows.append(_trial_training_row(art, lab, mp))
        meta = json.loads(mp.read_text(encoding="utf-8"))
        steps, tr, va = _steps_train_val_from_meta(meta)
        curves.append((lab, steps, tr, va))

    fig = plot_surrogate_training_curves(curves, out_path=out / "training_curves.png")
    close_fig(fig)

    report: dict = {"trials": training_rows, "field_eval": None}

    if not args.skip_field_eval:
        cfg = ProblemConfig()
        grid = GridConfig(nx=int(args.nx), nz=int(args.nz))
        sol = PDESolverConfig(max_outer_iters=int(args.pde_iters))
        t0 = time.perf_counter()
        fp = solve_pde_field(cfg, grid, sol)
        pde_wall = time.perf_counter() - t0
        fluid = ~building_mask(fp.x_grid, fp.z_grid, cfg)
        field_rows: list[dict] = []
        rmse_list: list[float] = []
        maxa_list: list[float] = []
        for art, lab in zip(artifacts, labels, strict=True):
            params, meta = load_surrogate_bundle(art)
            t1 = time.perf_counter()
            f_s = predict_field(
                params,
                cfg,
                grid,
                str(meta["activation"]),
                meta["scales"],
            )
            rt = time.perf_counter() - t1
            m0, m1 = metrics_rmse_max(f_s.T, fp.T, fluid)
            field_rows.append(
                {
                    "label": lab,
                    "rmse_vs_pde_degC": float(m0),
                    "max_abs_vs_pde_degC": float(m1),
                    "surrogate_runtime_s": float(rt),
                }
            )
            rmse_list.append(float(m0))
            maxa_list.append(float(m1))
        fig2 = plot_surrogate_field_errors(labels, rmse_list, maxa_list, out_path=out / "field_errors_vs_pde.png")
        close_fig(fig2)
        report["field_eval"] = {
            "nx": int(args.nx),
            "nz": int(args.nz),
            "pde_iters": int(args.pde_iters),
            "pde_runtime_s": float(pde_wall),
            "trials": field_rows,
        }

    (out / "trials_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_trials": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
