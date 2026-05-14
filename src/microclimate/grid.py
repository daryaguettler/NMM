"""structured grid, building mask, profiling helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from microclimate.config import GridConfig, ProblemConfig


def make_grid(cfg: ProblemConfig, grid_cfg: GridConfig) -> tuple[NDArray, NDArray]:
    """cell-centered x and z coordinates, shape (nx,) and (nz,)."""
    dx = (cfg.x_max - cfg.x_min) / grid_cfg.nx
    dz = (cfg.z_max - cfg.z_min) / grid_cfg.nz
    x = cfg.x_min + (np.arange(grid_cfg.nx) + 0.5) * dx
    z = cfg.z_min + (np.arange(grid_cfg.nz) + 0.5) * dz
    return x.astype(np.float64), z.astype(np.float64)


def cell_sizes(cfg: ProblemConfig, grid_cfg: GridConfig) -> tuple[float, float]:
    dx = (cfg.x_max - cfg.x_min) / grid_cfg.nx
    dz = (cfg.z_max - cfg.z_min) / grid_cfg.nz
    return dx, dz


def building_mask(
    x_grid: NDArray[np.floating[Any]],
    z_grid: NDArray[np.floating[Any]],
    cfg: ProblemConfig,
) -> NDArray[np.bool_]:
    """True inside building interior (cell centers)."""
    x2 = x_grid[:, np.newaxis]
    z2 = z_grid[np.newaxis, :]
    return (
        (x2 >= cfg.bldg_x_min)
        & (x2 <= cfg.bldg_x_max)
        & (z2 >= cfg.bldg_z_min)
        & (z2 <= cfg.bldg_z_max)
    )


def facade_adjacent_column(
    x_grid: NDArray[np.floating[Any]],
    z_grid: NDArray[np.floating[Any]],
    cfg: ProblemConfig,
    offset_m: float = 0.5,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Vertical profile at x ~ windward face + offset; returns (z_grid, T_column)."""
    target_x = cfg.bldg_x_min + offset_m
    ix = int(np.argmin(np.abs(x_grid - target_x)))
    col_z = z_grid.copy()
    return col_z, ix


def metrics_rmse_max(
    a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]], fluid: NDArray[np.bool_]
) -> tuple[float, float]:
    """Rmse and max abs error on fluid mask."""
    d = (a - b)[fluid]
    if d.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean(d * d)))
    mad = float(np.max(np.abs(d)))
    return rmse, mad
