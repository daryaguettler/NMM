"""lagrangian particle ensemble + grid binning for steady t(x,z)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from microclimate.config import GridConfig, ParticleConfig, ProblemConfig
from microclimate.grid import building_mask, cell_sizes, make_grid
from microclimate.types import TemperatureField


def _log_law_u_at_z(z: NDArray[np.floating[Any]], cfg: ProblemConfig) -> NDArray[np.floating[Any]]:
    z_ = np.maximum(np.asarray(z, dtype=np.float64), cfg.z_0 * 1.001)
    denom = np.log(cfg.z_ref / cfg.z_0)
    return cfg.U_ref * np.log(z_ / cfg.z_0) / denom


def _wind_interpolators(
    x: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    w: NDArray[np.floating[Any]],
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    # values[i,j] at x[i], z[j]
    ui = RegularGridInterpolator((x, z), u, bounds_error=False, fill_value=0.0)
    wi = RegularGridInterpolator((x, z), w, bounds_error=False, fill_value=0.0)
    return ui, wi


def _sample_mean_wind(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    cfg: ProblemConfig,
    ui: RegularGridInterpolator | None,
    wi: RegularGridInterpolator | None,
    use_pde: bool,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    pts = np.stack([px, pz], axis=1)
    if use_pde and ui is not None and wi is not None:
        um = ui(pts).astype(np.float64)
        wm = wi(pts).astype(np.float64)
        return um, wm
    um = _log_law_u_at_z(pz, cfg)
    return um, np.zeros_like(um)


def _escape_building_apply_thermal(
    px: NDArray[np.floating[Any]],
    pz: NDArray[np.floating[Any]],
    vx: NDArray[np.floating[Any]],
    vz: NDArray[np.floating[Any]],
    pT: NDArray[np.floating[Any]],
    cfg: ProblemConfig,
    pcfg: ParticleConfig,
) -> None:
    xmin, xmax = cfg.bldg_x_min, cfg.bldg_x_max
    zmin, zmax = cfg.bldg_z_min, cfg.bldg_z_max
    eps = 1e-3
    inside = (px >= xmin) & (px <= xmax) & (pz >= zmin) & (pz <= zmax)
    if not np.any(inside):
        return
    idx = np.where(inside)[0]
    px_i = px[idx]
    pz_i = pz[idx]
    d_left = px_i - xmin
    d_right = xmax - px_i
    d_bot = pz_i - zmin
    d_top = zmax - pz_i
    # face: 0 left, 1 right, 2 bottom, 3 top
    which = np.argmin(np.stack([d_left, d_right, d_bot, d_top], axis=1), axis=1)
    px_new = px_i.copy()
    pz_new = pz_i.copy()
    vx_new = vx[idx].copy()
    vz_new = vz[idx].copy()
    left = which == 0
    right = which == 1
    bottom = which == 2
    top = which == 3
    px_new[left] = xmin - eps
    vx_new[left] = -np.abs(vx_new[left]) * 0.85
    px_new[right] = xmax + eps
    vx_new[right] = np.abs(vx_new[right]) * 0.85
    pz_new[bottom] = zmin - eps
    vz_new[bottom] = -np.abs(vz_new[bottom]) * 0.85
    pz_new[top] = zmax + eps
    vz_new[top] = np.abs(vz_new[top]) * 0.85
    px[idx] = px_new
    pz[idx] = pz_new
    vx[idx] = vx_new
    vz[idx] = vz_new
    r_fac = float(pcfg.thermal_relax_facade)
    r_o = float(pcfg.thermal_relax_other)
    sub_l = idx[left]
    pT[sub_l] += r_fac * (cfg.T_facade_hot - pT[sub_l])
    sub_r = idx[right]
    pT[sub_r] += r_o * (cfg.T_ref - pT[sub_r])
    sub_b = idx[bottom]
    pT[sub_b] += r_o * (cfg.T_ground - pT[sub_b])
    sub_t = idx[top]
    pT[sub_t] += r_o * (cfg.T_ref - pT[sub_t])


def solve_particle_field(
    cfg: ProblemConfig,
    grid_cfg: GridConfig,
    particle_cfg: ParticleConfig,
    wind_field: TemperatureField | None = None,
) -> TemperatureField:
    """Steady binned t from particle ensemble; uses pde u,w when available."""
    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    dx, dz = cell_sizes(cfg, grid_cfg)
    nx, nz = int(grid_cfg.nx), int(grid_cfg.nz)
    solid = building_mask(x, z, cfg)
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

    n = int(particle_cfg.n_particles)
    px = rng.uniform(cfg.x_min + 1e-6, cfg.x_max - 1e-6, size=n).astype(np.float64)
    pz = rng.uniform(cfg.z_min + 1e-6, cfg.z_max - 1e-6, size=n).astype(np.float64)
    vx = np.zeros(n, dtype=np.float64)
    vz = np.zeros(n, dtype=np.float64)
    pT = np.full(n, cfg.T_ref, dtype=np.float64)
    um0, wm0 = _sample_mean_wind(px, pz, cfg, ui, wi, particle_cfg.use_pde_wind)
    vx[:] = um0
    vz[:] = wm0

    dt = float(particle_cfg.dt)
    T_L = max(float(particle_cfg.T_L), 1e-6)
    ti = float(particle_cfg.turbulent_intensity)
    vrel = float(particle_cfg.velocity_relax)

    sums = np.zeros((nx, nz), dtype=np.float64)
    counts = np.zeros((nx, nz), dtype=np.float64)

    n_spin = int(particle_cfg.n_steps_spinup)
    n_avg = int(particle_cfg.n_steps_average)
    total = n_spin + n_avg

    for step in range(total):
        um, wm = _sample_mean_wind(px, pz, cfg, ui, wi, particle_cfg.use_pde_wind)
        sigma = ti * np.maximum(np.abs(um), 0.4) * np.sqrt(dt / T_L)
        vx += sigma * rng.standard_normal(n)
        vz += sigma * rng.standard_normal(n)

        vx = (1.0 - vrel) * vx + vrel * um
        vz = (1.0 - vrel) * vz + vrel * wm

        vz += cfg.g * cfg.beta * (pT - cfg.T_ref) * dt

        px += vx * dt
        pz += vz * dt

        out_right = px > cfg.x_max
        if np.any(out_right):
            px[out_right] = cfg.x_min + 1e-3
            pz[out_right] = rng.uniform(cfg.z_min + 1e-3, cfg.z_max - 1e-3, size=int(np.sum(out_right)))
            pT[out_right] = cfg.T_ref
            pz_or = pz[out_right]
            vx[out_right] = _log_law_u_at_z(pz_or, cfg)
            vz[out_right] = 0.0

        px = np.clip(px, cfg.x_min + 1e-9, cfg.x_max - 1e-9)
        pz = np.clip(pz, cfg.z_min + 1e-9, cfg.z_max - 1e-9)

        left_out = px < cfg.x_min + 1e-6
        if np.any(left_out):
            pz_lo = pz[left_out]
            pT[left_out] = cfg.T_ref
            vx[left_out] = _log_law_u_at_z(pz_lo, cfg)
            vz[left_out] = 0.0
            px[left_out] = cfg.x_min + 1e-3

        _escape_building_apply_thermal(px, pz, vx, vz, pT, cfg, particle_cfg)

        if step >= n_spin:
            ix = np.floor((px - cfg.x_min) / dx).astype(np.int64)
            iz = np.floor((pz - cfg.z_min) / dz).astype(np.int64)
            ix = np.clip(ix, 0, nx - 1)
            iz = np.clip(iz, 0, nz - 1)
            np.add.at(sums, (ix, iz), pT)
            np.add.at(counts, (ix, iz), 1)

    T = np.full((nx, nz), np.nan, dtype=np.float64)
    fluid = ~solid
    cpos = counts > 0
    use = fluid & cpos
    T[use] = sums[use] / counts[use]
    T[fluid & ~cpos] = cfg.T_ref

    elapsed = time.perf_counter() - t0
    return TemperatureField(
        T=T,
        x_grid=x,
        z_grid=z,
        method_name="particles_lagrangian",
        runtime_seconds=float(elapsed),
        config=cfg,
    )
