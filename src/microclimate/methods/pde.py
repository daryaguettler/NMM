"""2d steady boussinesq reference: streamfunction-vorticity + temperature (numpy)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from microclimate.config import GridConfig, PDESolverConfig, ProblemConfig
from microclimate.grid import building_mask, cell_sizes, make_grid
from microclimate.types import TemperatureField

__all__ = ["solve_pde_field"]


def _log_law_u_at_nodes(z_1d: NDArray[np.floating[Any]], cfg: ProblemConfig) -> NDArray[np.floating[Any]]:
    z_ = np.maximum(z_1d.astype(np.float64), cfg.z_0 * 1.001)
    denom = np.log(cfg.z_ref / cfg.z_0)
    return cfg.U_ref * np.log(z_ / cfg.z_0) / denom


def _psi_inflow_column(z_1d: NDArray[np.floating[Any]], cfg: ProblemConfig) -> NDArray[np.floating[Any]]:
    u = _log_law_u_at_nodes(z_1d, cfg)
    psi = np.zeros_like(z_1d, dtype=np.float64)
    dz_col = float(z_1d[1] - z_1d[0]) if z_1d.size > 1 else 1.0
    psi[0] = 0.0
    for j in range(1, z_1d.size):
        psi[j] = psi[j - 1] + 0.5 * (u[j] + u[j - 1]) * dz_col
    return psi


def _apply_solid_temperatures(
    T: NDArray[np.floating[Any]],
    x: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
    cfg: ProblemConfig,
    dx: float,
    dz: float,
    solid: NDArray[np.bool_],
) -> None:
    x2 = x[:, np.newaxis]
    z2 = z[np.newaxis, :]
    windward = solid & (x2 <= cfg.bldg_x_min + 0.55 * dx) & (z2 <= cfg.bldg_z_max)
    dz0 = dz if z.size > 1 else 0.3
    ground_footprint = solid & (z2 <= cfg.bldg_z_min + 0.55 * dz0)
    T[:] = np.where(windward, cfg.T_facade_hot, T)
    T[:] = np.where(solid & ground_footprint & ~windward, cfg.T_ground, T)
    T[:] = np.where(solid & ~windward & ~ground_footprint, cfg.T_ref, T)


def solve_pde_field(
    cfg: ProblemConfig,
    grid_cfg: GridConfig,
    solver_cfg: PDESolverConfig | None = None,
) -> TemperatureField:
    """pseudo-time streamfunction-vorticity with temperature until t settles."""
    if solver_cfg is None:
        solver_cfg = PDESolverConfig()
    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    dx, dz = cell_sizes(cfg, grid_cfg)
    nx, nz = grid_cfg.nx, grid_cfg.nz
    solid = building_mask(x, z, cfg)
    fluid = ~solid

    nu = cfg.nu * float(solver_cfg.nu_scale)
    alpha = cfg.alpha * float(solver_cfg.alpha_scale)

    psi = np.zeros((nx, nz), dtype=np.float64)
    omega = np.zeros_like(psi)
    T = np.full((nx, nz), cfg.T_ref, dtype=np.float64)
    u = np.zeros_like(psi)
    w = np.zeros_like(psi)

    psi_left = _psi_inflow_column(z, cfg)
    frac_i = np.arange(nx, dtype=np.float64)[:, np.newaxis] / max(nx - 1, 1)
    psi[:] = psi_left[np.newaxis, :] * (1.0 - frac_i)

    def grad_T_x(Ta: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        g = np.zeros_like(Ta)
        g[1:-1, :] = (Ta[2:, :] - Ta[:-2, :]) / (2.0 * dx)
        g[0, :] = (Ta[1, :] - Ta[0, :]) / dx
        g[-1, :] = (Ta[-1, :] - Ta[-2, :]) / dx
        return g

    def laplacian(f: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        return (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 + (
            f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]
        ) / dz**2

    def upwind_adv_T(
        Ta: NDArray[np.floating[Any]], ua: NDArray[np.floating[Any]], wa: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        adv = np.zeros_like(Ta)
        i_s, i_e = 1, nx - 1
        j_s, j_e = 1, nz - 1
        Tu = Ta[i_s:i_e, j_s:j_e]
        uu = ua[i_s:i_e, j_s:j_e]
        u_pos = np.maximum(uu, 0.0)
        u_neg = np.minimum(uu, 0.0)
        Tx = (
            u_pos * (Tu - Ta[i_s - 1 : i_e - 1, j_s:j_e]) / dx
            + u_neg * (Ta[i_s + 1 : i_e + 1, j_s:j_e] - Tu) / dx
        )
        wu = wa[i_s:i_e, j_s:j_e]
        w_pos = np.maximum(wu, 0.0)
        w_neg = np.minimum(wu, 0.0)
        Tz = (
            w_pos * (Tu - Ta[i_s:i_e, j_s - 1 : j_e - 1]) / dz
            + w_neg * (Ta[i_s:i_e, j_s + 1 : j_e + 1] - Tu) / dz
        )
        adv[i_s:i_e, j_s:j_e] = Tx + Tz
        return adv

    def upwind_adv_omega(
        oa: NDArray[np.floating[Any]], ua: NDArray[np.floating[Any]], wa: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        adv = np.zeros_like(oa)
        i_s, i_e = 1, nx - 1
        j_s, j_e = 1, nz - 1
        Ou = oa[i_s:i_e, j_s:j_e]
        uu = ua[i_s:i_e, j_s:j_e]
        u_pos = np.maximum(uu, 0.0)
        u_neg = np.minimum(uu, 0.0)
        Ox = (
            u_pos * (Ou - oa[i_s - 1 : i_e - 1, j_s:j_e]) / dx
            + u_neg * (oa[i_s + 1 : i_e + 1, j_s:j_e] - Ou) / dx
        )
        wu = wa[i_s:i_e, j_s:j_e]
        w_pos = np.maximum(wu, 0.0)
        w_neg = np.minimum(wu, 0.0)
        Oz = (
            w_pos * (Ou - oa[i_s:i_e, j_s - 1 : j_e - 1]) / dz
            + w_neg * (oa[i_s:i_e, j_s + 1 : j_e + 1] - Ou) / dz
        )
        adv[i_s:i_e, j_s:j_e] = Ox + Oz
        return adv

    def velocities(
        ps: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        uo = np.zeros_like(ps)
        wo = np.zeros_like(ps)
        uo[:, 1:-1] = (ps[:, 2:] - ps[:, :-2]) / (2.0 * dz)
        uo[:, 0] = (ps[:, 1] - ps[:, 0]) / dz
        uo[:, -1] = (ps[:, -1] - ps[:, -2]) / dz
        uo[0, :] = _log_law_u_at_nodes(z, cfg)
        wo[1:-1, :] = -(ps[2:, :] - ps[:-2, :]) / (2.0 * dx)
        wo[0, :] = wo[1, :]
        wo[-1, :] = wo[-2, :]
        return uo, wo

    inv_den_psi = 1.0 / (2.0 / dx**2 + 2.0 / dz**2)
    u_top = float(_log_law_u_at_nodes(np.array([z[-1]], dtype=np.float64), cfg)[0])
    r_p = float(solver_cfg.relax_psi)

    prev_T = T.copy()
    prev_psi = psi.copy()
    for _it in range(solver_cfg.max_outer_iters):
        if solver_cfg.freeze_mean_wind_only:
            u[:, :] = 0.0
            for j in range(nz):
                u[:, j] = _log_law_u_at_nodes(np.array([z[j]]), cfg)[0]
            w[:, :] = 0.0
        else:
            u[:], w[:] = velocities(psi)

        u_mag = float(np.max(np.abs(u[fluid]) + np.abs(w[fluid])) + 1e-6)
        dt_t = min(
            0.4 * min(dx, dz) / max(u_mag, 1e-6),
            0.45 * min(dx**2, dz**2) / max(2.0 * alpha, 1e-12),
            float(solver_cfg.max_inner_T_adv),
        )

        # temperature: explicit pseudo-step + relaxation
        adv_T = upwind_adv_T(T, u, w)
        lap_T = np.zeros_like(T)
        lap_T[1:-1, 1:-1] = laplacian(T)
        rhs_T = -adv_T + alpha * lap_T
        T_trial = T + dt_t * rhs_T
        rT = float(solver_cfg.relax_T)
        T = (1.0 - rT) * T + rT * T_trial

        # boundary conditions on T
        T[0, :] = cfg.T_ref
        T[-1, :] = T[-2, :]
        T[:, -1] = T[:, -2]
        _apply_solid_temperatures(T, x, z, cfg, dx, dz, solid)
        j0 = 0
        T[:, j0] = np.where(fluid[:, j0], cfg.T_ground, T[:, j0])

        if not solver_cfg.freeze_mean_wind_only:
            dTdx = grad_T_x(T)
            adv_o = upwind_adv_omega(omega, u, w)
            lap_o = np.zeros_like(omega)
            lap_o[1:-1, 1:-1] = laplacian(omega)
            buoy = cfg.g * cfg.beta * dTdx
            rhs_o = -adv_o + nu * lap_o + buoy
            dt_o = min(
                0.35 * min(dx, dz) / max(u_mag, 1e-6),
                0.45 * min(dx**2, dz**2) / max(2.0 * nu, 1e-12),
                float(solver_cfg.max_inner_T_adv),
            )
            omega_trial = omega + dt_o * rhs_o
            r_o = float(solver_cfg.relax_omega)
            omega = (1.0 - r_o) * omega + r_o * omega_trial
            omega[solid] = 0.0

            for _sw in range(solver_cfg.poisson_sweeps):
                psi_in = psi
                pn = psi_in.copy()
                pn[solid] = 0.0
                lap_nbr = (pn[2:, 1:-1] + pn[:-2, 1:-1]) / dx**2 + (
                    pn[1:-1, 2:] + pn[1:-1, :-2]
                ) / dz**2
                psi_int = (lap_nbr + omega[1:-1, 1:-1]) * inv_den_psi
                psi_new = psi_in.copy()
                psi_new[1:-1, 1:-1] = np.where(fluid[1:-1, 1:-1], psi_int, 0.0)
                psi_new[0, :] = psi_left
                psi_new[-1, :] = psi_new[-2, :]
                psi_new[:, 0] = 0.0
                psi_new[:, -1] = psi_new[:, -2] + dz * u_top
                psi_new[solid] = 0.0
                psi = (1.0 - r_p) * psi_in + r_p * psi_new

        err_t = float(np.linalg.norm((T - prev_T)[fluid]) / (np.linalg.norm(T[fluid]) + 1e-9))
        err_p = float(np.linalg.norm((psi - prev_psi)[fluid]) / (np.linalg.norm(psi[fluid]) + 1e-9))
        prev_T = T.copy()
        prev_psi = psi.copy()
        if err_t < solver_cfg.tol_T and (
            solver_cfg.freeze_mean_wind_only or err_p < solver_cfg.tol_psi
        ):
            break

    T_out = T.copy()
    T_out[solid] = np.nan
    elapsed = time.perf_counter() - t0
    return TemperatureField(
        T=T_out.astype(np.float64),
        x_grid=x,
        z_grid=z,
        method_name="pde_streamfunction_vorticity",
        runtime_seconds=float(elapsed),
        config=cfg,
        u=u.astype(np.float64),
        w=w.astype(np.float64),
    )
