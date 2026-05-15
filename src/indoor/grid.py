"""grid utilities and solid masks for the indoor 2d domain."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor.config import IndoorGridConfig, IndoorProblemConfig


def make_grid(
    cfg: IndoorProblemConfig, grid_cfg: IndoorGridConfig
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """cell-centered x and z coordinates."""
    dx = (cfg.x_max - cfg.x_min) / grid_cfg.nx
    dz = (cfg.z_max - cfg.z_min) / grid_cfg.nz
    x = cfg.x_min + (np.arange(grid_cfg.nx) + 0.5) * dx
    z = cfg.z_min + (np.arange(grid_cfg.nz) + 0.5) * dz
    return x.astype(np.float64), z.astype(np.float64)


def cell_sizes(cfg: IndoorProblemConfig, grid_cfg: IndoorGridConfig) -> tuple[float, float]:
    dx = (cfg.x_max - cfg.x_min) / grid_cfg.nx
    dz = (cfg.z_max - cfg.z_min) / grid_cfg.nz
    return float(dx), float(dz)


def partition_mask(
    x: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    cfg: IndoorProblemConfig,
) -> NDArray[np.bool_]:
    """True where a cell is inside a solid partition header (above doorway).

    Each partition is one cell column wide; the doorway gap (z <= doorway_z_hi)
    is left open (fluid). The header (z > doorway_z_hi) is solid.
    """
    solid = np.zeros((x.size, z.size), dtype=bool)
    for xp in cfg.partition_x:
        col = int(np.argmin(np.abs(x - xp)))
        solid[col, :] = z > cfg.doorway_z_hi
    return solid


def room_index(
    x: NDArray[np.floating[Any]], cfg: IndoorProblemConfig
) -> NDArray[np.intp]:
    """Integer room id per x cell (0 = front, 1 = middle, 2 = back)."""
    pxs = sorted(cfg.partition_x)
    idx = np.zeros(x.size, dtype=np.intp)
    for i, xp in enumerate(pxs):
        idx[x > xp] = i + 1
    return idx


def metrics_rmse_max(
    a: NDArray[np.floating[Any]],
    b: NDArray[np.floating[Any]],
    fluid: NDArray[np.bool_],
) -> tuple[float, float]:
    d = (a - b)[fluid]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))
