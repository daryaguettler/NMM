"""lagrangian particle ensemble for a closed indoor room.

Key differences from the outdoor solver:
  - Closed domain: no periodic reinject.  Windows inject/eject particles when open.
  - Partition walls: specular reflection; adiabatic (no thermal relaxation).
  - Mean wind from PDE velocity field when available; otherwise zero (closed room).
  - EXPECTED FAILURE: without pressure coupling, particles cannot form the closed
    convection cell driven by buoyancy.  They diffuse and rise along the hot wall
    but cannot sustain the return flow across the ceiling and down the cool wall.
    This is by design — the failure mode demonstrates when Lagrangian methods
    require momentum closure.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from indoor.config import IndoorGridConfig, IndoorParticleConfig, IndoorProblemConfig
from indoor.grid import cell_sizes, make_grid, partition_mask
from indoor.types import IndoorField

__all__ = ["solve_particle_field", "sample_particle_trajectories"]


# ── wind interpolation from PDE field ─────────────────────────────────────────

def _wind_interpolators(
    xg: NDArray[np.floating[Any]],
    zg: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    w: NDArray[np.floating[Any]],
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    ui = RegularGridInterpolator((xg, zg), u, bounds_error=False, fill_value=0.0)
    wi = RegularGridInterpolator((xg, zg), w, bounds_error=False, fill_value=0.0)
    return ui, wi


def _sample_wind(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    ui: RegularGridInterpolator | None,
    wi: RegularGridInterpolator | None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    if ui is not None and wi is not None:
        pts = np.stack([px, pz], axis=1)
        return ui(pts).astype(np.float64), wi(pts).astype(np.float64)
    return np.zeros_like(px), np.zeros_like(pz)


# ── partition collision detection ─────────────────────────────────────────────

def _partition_x_cols(
    x: NDArray[np.floating[Any]], cfg: IndoorProblemConfig
) -> list[float]:
    """x-coordinates of partition cell centres (one per partition)."""
    return [float(x[int(np.argmin(np.abs(x - xp)))]) for xp in cfg.partition_x]


def _reflect_partitions(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    vx: NDArray[np.floating[Any]],
    part_cols: list[float],
    dx: float,
    cfg: IndoorProblemConfig,
) -> None:
    """Reflect particles off partition solid cells; doorway is passable."""
    half = dx / 2.0
    for xp in part_cols:
        near = np.abs(px - xp) < half
        if not np.any(near):
            continue
        above_door = pz[near] > cfg.doorway_z_hi
        idx = np.where(near)[0][above_door]   # indices hitting solid part
        if idx.size == 0:
            continue
        # reflect: push out and flip x-velocity
        side = np.sign(px[idx] - xp)
        px[idx] = xp + side * (half + 1e-4)
        vx[idx] = side * np.abs(vx[idx]) * 0.85


# ── wall boundary handler ─────────────────────────────────────────────────────

def _apply_boundaries(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    vx: NDArray[np.floating[Any]],
    vz: NDArray[np.floating[Any]],
    pT: NDArray[np.floating[Any]],
    cfg: IndoorProblemConfig,
    pcfg: IndoorParticleConfig,
    rng: np.random.Generator,
) -> None:
    # ── floor ────────────────────────────────────────────────────────────────
    below = pz < cfg.z_min
    if np.any(below):
        pz[below] = 2.0 * cfg.z_min - pz[below]
        vz[below] = np.abs(vz[below]) * 0.85
        pT[below] += pcfg.relax_floor * (cfg.T_floor - pT[below])

    # ── ceiling ───────────────────────────────────────────────────────────────
    above = pz > cfg.z_max
    if np.any(above):
        pz[above] = 2.0 * cfg.z_max - pz[above]
        vz[above] = -np.abs(vz[above]) * 0.85
        pT[above] += pcfg.relax_ceiling * (cfg.T_ceiling - pT[above])

    # ── left wall (x = x_min) ────────────────────────────────────────────────
    left = px < cfg.x_min
    if np.any(left):
        idx = np.where(left)[0]
        win_s = (pz[idx] >= cfg.win_south_z_lo) & (pz[idx] <= cfg.win_south_z_hi)
        # window: reinject from inlet when open
        win_idx = idx[win_s & (cfg.window_open > 0.0)]
        if win_idx.size > 0:
            px[win_idx] = cfg.x_min + 1e-4
            vx[win_idx] = cfg.U_window * cfg.window_open
            vz[win_idx] = 0.0
            pT[win_idx] = cfg.T_outdoor
        # solid wall: reflect + hot-facade BC
        solid_idx = idx[~win_s | (cfg.window_open == 0.0)]
        if solid_idx.size > 0:
            px[solid_idx] = 2.0 * cfg.x_min - px[solid_idx]
            vx[solid_idx] = np.abs(vx[solid_idx]) * 0.85
            pT[solid_idx] += pcfg.relax_hot_wall * (cfg.T_facade_hot - pT[solid_idx])

    # ── right wall (x = x_max) ───────────────────────────────────────────────
    right = px > cfg.x_max
    if np.any(right):
        idx = np.where(right)[0]
        win_n = (pz[idx] >= cfg.win_north_z_lo) & (pz[idx] <= cfg.win_north_z_hi)
        # outlet window: respawn at inlet window
        out_idx = idx[win_n & (cfg.window_open > 0.0)]
        if out_idx.size > 0:
            px[out_idx] = cfg.x_min + 1e-4
            pz[out_idx] = rng.uniform(
                cfg.win_south_z_lo, cfg.win_south_z_hi, size=out_idx.size
            )
            vx[out_idx] = cfg.U_window * cfg.window_open
            vz[out_idx] = 0.0
            pT[out_idx] = cfg.T_outdoor
        # solid wall: reflect + cool-facade BC
        solid_idx = idx[~win_n | (cfg.window_open == 0.0)]
        if solid_idx.size > 0:
            px[solid_idx] = 2.0 * cfg.x_max - px[solid_idx]
            vx[solid_idx] = -np.abs(vx[solid_idx]) * 0.85
            pT[solid_idx] += pcfg.relax_cool_wall * (cfg.T_facade_cool - pT[solid_idx])


# ── one integration step ──────────────────────────────────────────────────────

def _step(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    vx: NDArray[np.floating[Any]],
    vz: NDArray[np.floating[Any]],
    pT: NDArray[np.floating[Any]],
    cfg: IndoorProblemConfig,
    pcfg: IndoorParticleConfig,
    rng: np.random.Generator,
    ui: RegularGridInterpolator | None,
    wi: RegularGridInterpolator | None,
    part_cols: list[float],
    dx: float,
) -> None:
    um, wm = _sample_wind(px, pz, ui, wi)
    dt   = float(pcfg.dt)
    T_L  = max(float(pcfg.T_L), 1e-6)
    ti   = float(pcfg.turbulent_intensity)
    vrel = float(pcfg.velocity_relax)

    u_ref = np.maximum(np.abs(um), 0.2)
    sigma = ti * u_ref * np.sqrt(dt / T_L)
    vx += sigma * rng.standard_normal(px.size)
    vz += sigma * rng.standard_normal(px.size)

    vx = (1.0 - vrel) * vx + vrel * um
    vz = (1.0 - vrel) * vz + vrel * wm

    # buoyancy
    vz += cfg.g * cfg.beta * (pT - cfg.T_ref) * dt

    px += vx * dt
    pz += vz * dt

    _reflect_partitions(px, pz, vx, part_cols, dx, cfg)
    _apply_boundaries(px, pz, vx, vz, pT, cfg, pcfg, rng)

    px[:] = np.clip(px, cfg.x_min, cfg.x_max)
    pz[:] = np.clip(pz, cfg.z_min, cfg.z_max)


# ── public API ────────────────────────────────────────────────────────────────

def solve_particle_field(
    cfg: IndoorProblemConfig,
    grid_cfg: IndoorGridConfig,
    particle_cfg: IndoorParticleConfig,
    wind_field: IndoorField | None = None,
) -> IndoorField:
    """Steady binned T from particle ensemble."""
    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    dx, dz = cell_sizes(cfg, grid_cfg)
    nx, nz = x.size, z.size
    solid = partition_mask(x, z, cfg)
    rng = np.random.default_rng(int(particle_cfg.seed))

    ui = wi = None
    if (
        particle_cfg.use_pde_wind
        and wind_field is not None
        and wind_field.u is not None
        and wind_field.w is not None
    ):
        ui, wi = _wind_interpolators(
            np.asarray(wind_field.x_grid),
            np.asarray(wind_field.z_grid),
            np.asarray(wind_field.u),
            np.asarray(wind_field.w),
        )

    part_cols = _partition_x_cols(x, cfg)
    n = int(particle_cfg.n_particles)
    px = rng.uniform(cfg.x_min + 1e-3, cfg.x_max - 1e-3, n)
    pz = rng.uniform(cfg.z_min + 1e-3, cfg.z_max - 1e-3, n)
    vx = np.zeros(n, dtype=np.float64)
    vz = np.zeros(n, dtype=np.float64)
    pT = np.full(n, cfg.T_ref, dtype=np.float64)

    sums   = np.zeros((nx, nz), dtype=np.float64)
    counts = np.zeros((nx, nz), dtype=np.float64)

    n_spin = int(particle_cfg.n_steps_spinup)
    n_avg  = int(particle_cfg.n_steps_average)

    for step in range(n_spin + n_avg):
        _step(px, pz, vx, vz, pT, cfg, particle_cfg, rng, ui, wi, part_cols, dx)
        if step >= n_spin:
            ix = np.clip(np.floor((px - cfg.x_min) / dx).astype(np.int64), 0, nx - 1)
            iz = np.clip(np.floor((pz - cfg.z_min) / dz).astype(np.int64), 0, nz - 1)
            np.add.at(sums,   (ix, iz), pT)
            np.add.at(counts, (ix, iz), 1)

    T = np.where((counts > 0) & ~solid, sums / np.maximum(counts, 1), np.nan)
    T[~solid & (counts == 0)] = cfg.T_ref

    elapsed = time.perf_counter() - t0
    return IndoorField(
        T=T,
        x_grid=x,
        z_grid=z,
        method_name="particles_indoor",
        runtime_seconds=float(elapsed),
        config=cfg,
    )


def sample_particle_trajectories(
    cfg: IndoorProblemConfig,
    grid_cfg: IndoorGridConfig,
    particle_cfg: IndoorParticleConfig,
    wind_field: IndoorField | None,
    *,
    n_trace: int,
    spinup_steps: int,
    record_steps: int,
    record_stride: int = 5,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Same integration as solve_particle_field; records a subset for animation."""
    x, z = make_grid(cfg, grid_cfg)
    dx, _ = cell_sizes(cfg, grid_cfg)
    rng = np.random.default_rng(int(particle_cfg.seed))

    ui = wi = None
    if (
        particle_cfg.use_pde_wind
        and wind_field is not None
        and wind_field.u is not None
        and wind_field.w is not None
    ):
        ui, wi = _wind_interpolators(
            np.asarray(wind_field.x_grid),
            np.asarray(wind_field.z_grid),
            np.asarray(wind_field.u),
            np.asarray(wind_field.w),
        )

    part_cols = _partition_x_cols(x, cfg)
    n = int(n_trace)
    px = rng.uniform(cfg.x_min + 1e-3, cfg.x_max - 1e-3, n)
    pz = rng.uniform(cfg.z_min + 1e-3, cfg.z_max - 1e-3, n)
    vx = np.zeros(n, dtype=np.float64)
    vz = np.zeros(n, dtype=np.float64)
    pT = np.full(n, cfg.T_ref, dtype=np.float64)

    rs      = max(1, int(record_stride))
    n_snaps = (int(record_steps) + rs - 1) // rs
    out_x   = np.zeros((n_snaps, n), dtype=np.float64)
    out_z   = np.zeros((n_snaps, n), dtype=np.float64)
    out_T   = np.zeros((n_snaps, n), dtype=np.float64)
    snap_i  = 0
    total   = int(spinup_steps) + int(record_steps)

    for step in range(total):
        _step(px, pz, vx, vz, pT, cfg, particle_cfg, rng, ui, wi, part_cols, dx)
        if step >= int(spinup_steps):
            off = step - int(spinup_steps)
            if off % rs == 0 and snap_i < n_snaps:
                out_x[snap_i] = px.copy()
                out_z[snap_i] = pz.copy()
                out_T[snap_i] = pT.copy()
                snap_i += 1

    t_snap = np.arange(snap_i, dtype=np.float64) * rs * particle_cfg.dt
    return {
        "t":  t_snap,
        "px": out_x[:snap_i],
        "pz": out_z[:snap_i],
        "pT": out_T[:snap_i],
    }
