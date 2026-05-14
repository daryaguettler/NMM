"""plotting helpers for microclimate comparison figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from microclimate.config import ProblemConfig
from microclimate.types import TemperatureField


def _common_temp_limits(fields: Sequence[TemperatureField]) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []
    for f in fields:
        tt = np.asarray(f.T)
        q = tt[np.isfinite(tt)]
        if q.size:
            mins.append(float(np.min(q)))
            maxs.append(float(np.max(q)))
    if not mins:
        return 20.0, 55.0
    return min(mins) - 0.5, max(maxs) + 0.5


def plot_temperature_panel(
    ax: plt.Axes,
    field: TemperatureField,
    cfg: ProblemConfig,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    xg = np.asarray(field.x_grid)
    zg = np.asarray(field.z_grid)
    X, Z = np.meshgrid(xg, zg, indexing="ij")
    Tm = np.ma.masked_invalid(np.asarray(field.T))
    cm = ax.pcolormesh(X, Z, Tm, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
    ax.add_patch(
        mpatches.Rectangle(
            (cfg.bldg_x_min, cfg.bldg_z_min),
            cfg.bldg_x_max - cfg.bldg_x_min,
            cfg.bldg_z_max - cfg.bldg_z_min,
            fill=True,
            facecolor="0.45",
            edgecolor="k",
            linewidth=0.6,
            zorder=5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (cfg.bldg_x_min, cfg.bldg_z_min),
            0.15,
            cfg.bldg_z_max - cfg.bldg_z_min,
            fill=True,
            facecolor="crimson",
            edgecolor="none",
            zorder=6,
        )
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    plt.colorbar(cm, ax=ax, fraction=0.046, pad=0.04, label="T (°C)")


def plot_method_grid(
    fields: Sequence[TemperatureField],
    titles: Sequence[str],
    cfg: ProblemConfig,
    out_path: Path | None = None,
) -> plt.Figure:
    n = len(fields)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    vmin, vmax = _common_temp_limits(fields)
    for i, f in enumerate(fields):
        plot_temperature_panel(axes_flat[i], f, cfg, vmin, vmax, titles[i])
    for j in range(len(fields), len(axes_flat)):
        axes_flat[j].set_visible(False)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig


def plot_vertical_profiles(
    z: np.ndarray,
    series: dict[str, np.ndarray],
    cfg: ProblemConfig,
    out_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 5.5), constrained_layout=True)
    for label, vals in series.items():
        ax.plot(vals, z, label=label, linewidth=1.6)
    ax.axhline(cfg.bldg_z_max, color="0.5", linewidth=0.8, linestyle="--", label="roof")
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="best", fontsize=9)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig


def facade_profile(field: TemperatureField, cfg: ProblemConfig, offset_m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (z, t_at_column) for nearest x to windward face + offset (into fluid)."""
    x = np.asarray(field.x_grid)
    z = np.asarray(field.z_grid)
    target = cfg.bldg_x_min + offset_m
    ix = int(np.argmin(np.abs(x - target)))
    col = np.asarray(field.T[ix, :])
    m = building_mask_simple(x[ix], z, cfg)
    col = np.where(m, np.nan, col)
    return z, col


def building_mask_simple(x_val: float, z_1d: np.ndarray, cfg: ProblemConfig) -> np.ndarray:
    return (
        (x_val >= cfg.bldg_x_min)
        & (x_val <= cfg.bldg_x_max)
        & (z_1d >= cfg.bldg_z_min)
        & (z_1d <= cfg.bldg_z_max)
    )


def close_fig(fig: plt.Figure) -> None:
    plt.close(fig)
