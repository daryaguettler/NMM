"""closed-form zonified temperature (röckle-style baseline)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microclimate.config import GridConfig, ProblemConfig
from microclimate.grid import building_mask, make_grid
from microclimate.types import TemperatureField


def analytic_temperature(
    x: NDArray[np.floating[Any]] | float,
    z: NDArray[np.floating[Any]] | float,
    cfg: ProblemConfig,
) -> NDArray[np.floating[Any]] | float:
    """Piecewise zonified t(x,z); plume is modeled in fluid immediately upwind of facade."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    T = np.full_like(x, cfg.T_ref, dtype=np.float64)

    fluid_upwind = (x < cfg.bldg_x_min) | (x > cfg.bldg_x_max) | (z > cfg.bldg_z_max)

    # far upwind
    up_far = x < 20.0
    T = np.where(up_far & fluid_upwind, cfg.T_ref, T)

    # displacement layer: no thermal anomaly in spec
    disp = (x >= 20.0) & (x < cfg.bldg_x_min)
    T = np.where(disp, cfg.T_ref, T)

    # windward plume: incoming wind at T_ref suppresses the thermal BL on the
    # windward face — PDE validation shows fluid west of facade stays within
    # ~0.35°C of ambient, so no plume excess is applied here.

    # wake carryover (lee of building) — amplitude calibrated to PDE: ~+2.4°C peak
    lee_start = cfg.bldg_x_max
    lee_end = 50.0
    in_wake = (x > lee_start) & (x <= lee_end)
    x_peak = cfg.bldg_x_max + 5.0
    sigma = 6.0
    wake_shape = np.exp(-0.5 * np.square((x - x_peak) / sigma))
    z_scale = np.clip((cfg.z_max - z) / cfg.z_max, 0.0, 1.0)
    t_hat = (cfg.T_facade_hot - cfg.T_ref) * 0.10
    T = np.where(in_wake, cfg.T_ref + t_hat * wake_shape * z_scale, T)

    # far downwind relaxation
    far = x > lee_end
    relax = np.clip((x - lee_end) / 10.0, 0.0, 1.0)  # blend over 10 m
    T = np.where(far, cfg.T_ref + (T - cfg.T_ref) * (1.0 - relax), T)

    # above-building mixed zone: keep weak lee signature if downstream
    above = (x >= cfg.bldg_x_min) & (x <= cfg.bldg_x_max) & (z > cfg.bldg_z_max)
    T = np.where(above, cfg.T_ref + 0.15 * t_hat * wake_shape * z_scale, T)

    scalar = x.ndim == 0 and z.ndim == 0
    return float(T) if scalar else T


def solve_analytic_field(cfg: ProblemConfig, grid_cfg: GridConfig) -> TemperatureField:
    """Rasterize analytic_temperature on the configured grid."""
    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    x2, z2 = np.meshgrid(x, z, indexing="ij")
    T = analytic_temperature(x2, z2, cfg)
    mask = building_mask(x, z, cfg)
    T = np.where(mask, np.nan, T)
    j0 = 0
    fluid0 = ~mask[:, j0]
    T[:, j0] = np.where(fluid0, cfg.T_ground, np.nan)
    elapsed = time.perf_counter() - t0
    return TemperatureField(
        T=T.astype(np.float64),
        x_grid=x,
        z_grid=z,
        method_name="analytic_zones",
        runtime_seconds=float(elapsed),
        config=cfg,
    )
