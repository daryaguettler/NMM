"""two-layer displacement-ventilation analytic model (Linden-style, three rooms).

For each room we predict:
  - a lower cool layer at T_lower (near-outdoor / inlet temperature)
  - an upper warm layer at T_upper (heated by the hot facade)
  - an interface height h_int separating them

Rooms are coupled through doorways: the temperature entering the next room is
the mean of the sending room's two layers, weighted by the fraction of the
doorway area in each layer.

Reference: Linden, Lane-Serff & Smeed (1990) JFM 212; Linden (1999) Ann Rev
Fluid Mech 31.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor.config import IndoorGridConfig, IndoorProblemConfig
from indoor.grid import make_grid, partition_mask, room_index
from indoor.types import IndoorField

__all__ = ["solve_analytic_field"]

_K_AIR = 0.026    # W/(m·K)
_RHO_CP = 1.2 * 1005.0  # J/(m³·K)


def _nusselt(Ra: float) -> float:
    """Nusselt number for natural convection on a vertical heated wall."""
    if Ra < 1e4:
        return 1.0
    if Ra < 1e9:
        return 0.22 * Ra**0.28       # laminar / transitional
    return 0.046 * Ra ** (1.0 / 3.0)  # turbulent


def _room_two_layer(
    *,
    T_hot: float,
    T_cool: float,
    T_outdoor: float,
    window_open: float,
    U_window: float,
    h_window: float,
    H: float,
    cfg: IndoorProblemConfig,
) -> tuple[float, float, float]:
    """Compute (h_interface, T_lower, T_upper) for one room.

    Args:
        T_hot: temperature of the warm bounding surface (hot facade or doorway inlet).
        T_cool: temperature of the cool bounding surface (cool facade or adiabatic).
        T_outdoor: outdoor inlet air temperature.
        window_open: 0–1 ventilation factor.
        U_window: inflow speed when open (m/s).
        h_window: window opening height span (m).
        H: room height (m).
        cfg: physical constants.
    """
    dT = abs(T_hot - T_cool)
    Ra_H = cfg.g * cfg.beta * max(dT, 0.1) * H**3 / (cfg.nu * cfg.alpha)
    Nu = _nusselt(Ra_H)
    h_c = Nu * _K_AIR / H  # convective heat transfer coefficient (W/m²K)
    T_mean_wall = (T_hot + T_cool) / 2.0
    Q_wall = h_c * H * abs(T_hot - T_mean_wall)  # heat flux per metre depth

    if window_open > 0.0 and U_window > 0.0:
        # Displacement ventilation: cool outdoor air enters at bottom of window,
        # warm air exits from top.  Upper layer temperature from heat balance.
        Q_v = U_window * window_open * h_window  # volume flux (m²/s per m depth)
        dT_rise = Q_wall / (_RHO_CP * Q_v)
        T_upper = T_outdoor + dT_rise
        T_lower = T_outdoor
        # Interface height: thermal pressure drives the stratification.
        # Simplified: interface sits near mid-window height.
        h_int = (cfg.win_south_z_lo + cfg.win_south_z_hi) / 2.0
    else:
        # Closed room: buoyancy builds up a stratified steady state.
        # Upper layer ≈ 35–40 % of wall-to-reference dT above T_cool.
        T_upper = T_cool + 0.38 * dT
        T_lower = T_cool + 0.08 * dT
        h_int = H / 2.0   # interface at mid-height for closed room

    return float(h_int), float(T_lower), float(T_upper)


def _doorway_outlet_T(
    T_lower: float,
    T_upper: float,
    h_int: float,
    cfg: IndoorProblemConfig,
) -> float:
    """Bulk temperature entering the next room through the doorway.

    The doorway spans z ∈ [doorway_z_lo, doorway_z_hi].  We weight each layer
    by the fraction of doorway height it occupies.
    """
    dz_lo = cfg.doorway_z_hi  # doorway bottom sits at z=0
    dz_hi = cfg.doorway_z_hi
    h_lo = min(h_int, dz_hi) - cfg.doorway_z_lo   # lower-layer height in doorway
    h_hi = dz_hi - max(h_int, cfg.doorway_z_lo)    # upper-layer height in doorway
    h_lo = max(h_lo, 0.0)
    h_hi = max(h_hi, 0.0)
    total = h_lo + h_hi
    if total < 1e-6:
        return (T_lower + T_upper) / 2.0
    return (h_lo * T_lower + h_hi * T_upper) / total


def _project_room(
    T_field: NDArray[np.floating[Any]],
    x: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    x_lo: float,
    x_hi: float,
    T_left: float,
    T_right: float,
    h_int: float,
    T_lower: float,
    T_upper: float,
) -> None:
    """Write two-layer temperature onto cells inside one room (in-place)."""
    in_room = (x >= x_lo) & (x <= x_hi)
    W_room = x_hi - x_lo
    for i in np.where(in_room)[0]:
        frac = (x[i] - x_lo) / max(W_room, 1e-6)
        # horizontal gradient blends T_left → T_right at each layer
        T_lo_x = T_lower + (T_right - T_left) * frac * 0.5
        T_hi_x = T_upper - (T_upper - T_lower) * frac * 0.15
        for j in range(z.size):
            T_field[i, j] = T_hi_x if z[j] > h_int else T_lo_x


def solve_analytic_field(
    cfg: IndoorProblemConfig,
    grid_cfg: IndoorGridConfig,
) -> IndoorField:
    """Rasterise the two-layer model onto the configured grid."""
    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    solid = partition_mask(x, z, cfg)
    nx, nz = x.size, z.size
    T = np.full((nx, nz), cfg.T_ref, dtype=np.float64)

    h_window = cfg.win_south_z_hi - cfg.win_south_z_lo

    pxs = sorted(cfg.partition_x)
    room_bounds = [
        (cfg.x_min, pxs[0]),
        (pxs[0],    pxs[1]),
        (pxs[1],    cfg.x_max),
    ]

    # Room 1 (front): hot left facade → first partition
    h1, Tlo1, Tup1 = _room_two_layer(
        T_hot=cfg.T_facade_hot,
        T_cool=cfg.T_ref,
        T_outdoor=cfg.T_outdoor,
        window_open=cfg.window_open,
        U_window=cfg.U_window,
        h_window=h_window,
        H=cfg.z_max - cfg.z_min,
        cfg=cfg,
    )
    _project_room(T, x, z, *room_bounds[0], cfg.T_facade_hot, cfg.T_ref, h1, Tlo1, Tup1)

    # coupling: temperature delivered to Room 2 through the doorway
    T_in2 = _doorway_outlet_T(Tlo1, Tup1, h1, cfg)

    # Room 2 (middle): air from Room 1 on left, flows toward Room 3
    h2, Tlo2, Tup2 = _room_two_layer(
        T_hot=T_in2,
        T_cool=cfg.T_ref,
        T_outdoor=cfg.T_outdoor,
        window_open=0.0,   # no direct window in middle room
        U_window=cfg.U_window,
        h_window=h_window,
        H=cfg.z_max - cfg.z_min,
        cfg=cfg,
    )
    _project_room(T, x, z, *room_bounds[1], T_in2, cfg.T_ref, h2, Tlo2, Tup2)

    T_in3 = _doorway_outlet_T(Tlo2, Tup2, h2, cfg)

    # Room 3 (back): air from Room 2 on left, cool north facade on right
    h3, Tlo3, Tup3 = _room_two_layer(
        T_hot=T_in3,
        T_cool=cfg.T_facade_cool,
        T_outdoor=cfg.T_outdoor,
        window_open=0.0,
        U_window=cfg.U_window,
        h_window=h_window,
        H=cfg.z_max - cfg.z_min,
        cfg=cfg,
    )
    _project_room(T, x, z, *room_bounds[2], T_in3, cfg.T_facade_cool, h3, Tlo3, Tup3)

    # solid partition headers: NaN (not physical air)
    T_out = np.where(solid, np.nan, T)

    elapsed = time.perf_counter() - t0
    return IndoorField(
        T=T_out,
        x_grid=x,
        z_grid=z,
        method_name="analytic_two_layer",
        runtime_seconds=float(elapsed),
        config=cfg,
    )
