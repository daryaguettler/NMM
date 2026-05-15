"""2d steady boussinesq for a closed indoor room: streamfunction-vorticity.

Key differences from the outdoor solver (microclimate/methods/pde.py):
  - ψ = 0 on all four walls for closed room; inflow profile when windows open
  - Ceiling BC is T_ceiling (Dirichlet), not zero-gradient
  - Left wall is T_facade_hot; right wall is T_facade_cool
  - Interior partition mask: solid cells with adiabatic temperature (no
    Dirichlet override — diffusion drives T toward neighbour average = dT/dn≈0)
  - No log-law wind anywhere
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from indoor.config import IndoorGridConfig, IndoorPDESolverConfig, IndoorProblemConfig
from indoor.grid import cell_sizes, make_grid, partition_mask
from indoor.types import IndoorField

__all__ = ["solve_pde_field"]


# ── streamfunction boundary profiles ──────────────────────────────────────────

def _psi_left(z: NDArray[np.floating[Any]], cfg: IndoorProblemConfig) -> NDArray[np.floating[Any]]:
    """ψ on the left wall: 0 if closed, inflow ramp if window open."""
    Q = cfg.U_window * cfg.window_open * (cfg.win_south_z_hi - cfg.win_south_z_lo)
    psi = np.zeros_like(z)
    in_win = (z >= cfg.win_south_z_lo) & (z <= cfg.win_south_z_hi)
    above_win = z > cfg.win_south_z_hi
    psi[in_win] = cfg.U_window * cfg.window_open * (z[in_win] - cfg.win_south_z_lo)
    psi[above_win] = Q
    return psi


def _psi_top(cfg: IndoorProblemConfig) -> float:
    """ψ on the ceiling: equals total window flux (conserved along streamline)."""
    return float(cfg.U_window * cfg.window_open * (cfg.win_south_z_hi - cfg.win_south_z_lo))


# ── differential operators ─────────────────────────────────────────────────────

def _laplacian(
    f: NDArray[np.floating[Any]], dx: float, dz: float
) -> NDArray[np.floating[Any]]:
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        + (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dz**2
    )
    return lap


def _grad_x(
    f: NDArray[np.floating[Any]], dx: float
) -> NDArray[np.floating[Any]]:
    g = np.zeros_like(f)
    g[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dx)
    g[0, :] = (f[1, :] - f[0, :]) / dx
    g[-1, :] = (f[-1, :] - f[-2, :]) / dx
    return g


def _upwind_adv(
    f: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    w: NDArray[np.floating[Any]],
    dx: float,
    dz: float,
) -> NDArray[np.floating[Any]]:
    adv = np.zeros_like(f)
    i0, i1 = 1, f.shape[0] - 1
    j0, j1 = 1, f.shape[1] - 1
    fu = f[i0:i1, j0:j1]
    uu = u[i0:i1, j0:j1]
    wu = w[i0:i1, j0:j1]
    adv[i0:i1, j0:j1] = (
        np.maximum(uu, 0.0) * (fu - f[i0-1:i1-1, j0:j1]) / dx
        + np.minimum(uu, 0.0) * (f[i0+1:i1+1, j0:j1] - fu) / dx
        + np.maximum(wu, 0.0) * (fu - f[i0:i1, j0-1:j1-1]) / dz
        + np.minimum(wu, 0.0) * (f[i0:i1, j0+1:j1+1] - fu) / dz
    )
    return adv


def _velocities(
    psi: NDArray[np.floating[Any]], dx: float, dz: float
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    u = np.zeros_like(psi)
    w = np.zeros_like(psi)
    # u = dψ/dz
    u[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dz)
    u[:, 0]    = (psi[:, 1]  - psi[:, 0])   / dz
    u[:, -1]   = (psi[:, -1] - psi[:, -2])  / dz
    # w = -dψ/dx
    w[1:-1, :] = -(psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
    w[0, :]    = 0.0          # left wall: no normal flow
    w[-1, :]   = w[-2, :]     # right wall: zero-gradient outflow
    return u, w


# ── main solver ───────────────────────────────────────────────────────────────

def solve_pde_field(
    cfg: IndoorProblemConfig,
    grid_cfg: IndoorGridConfig,
    solver_cfg: IndoorPDESolverConfig | None = None,
) -> IndoorField:
    """Pseudo-time streamfunction-vorticity with temperature until convergence."""
    if solver_cfg is None:
        solver_cfg = IndoorPDESolverConfig()

    t0 = time.perf_counter()
    x, z = make_grid(cfg, grid_cfg)
    dx, dz = cell_sizes(cfg, grid_cfg)
    nx, nz = x.size, z.size
    solid = partition_mask(x, z, cfg)
    fluid = ~solid

    nu    = cfg.nu    * float(solver_cfg.nu_scale)
    alpha = cfg.alpha * float(solver_cfg.alpha_scale)

    # ── state variables ───────────────────────────────────────────────────────
    psi   = np.zeros((nx, nz), dtype=np.float64)
    omega = np.zeros((nx, nz), dtype=np.float64)
    # initialise T with a linear horizontal gradient (hot left → cool right)
    frac_x = (x[:, np.newaxis] - cfg.x_min) / max(cfg.x_max - cfg.x_min, 1e-6)
    T = cfg.T_facade_hot * (1.0 - frac_x) + cfg.T_facade_cool * frac_x
    T = np.broadcast_to(T, (nx, nz)).copy()
    u = np.zeros((nx, nz), dtype=np.float64)
    w = np.zeros_like(u)

    # ── pre-computed BCs ──────────────────────────────────────────────────────
    psi_left_bc  = _psi_left(z, cfg)
    psi_top_bc   = _psi_top(cfg)
    inv_den_psi  = 1.0 / (2.0 / dx**2 + 2.0 / dz**2)
    r_p = float(solver_cfg.relax_psi)

    # window z-masks for T BCs
    win_s = (z >= cfg.win_south_z_lo) & (z <= cfg.win_south_z_hi)
    win_n = (z >= cfg.win_north_z_lo) & (z <= cfg.win_north_z_hi)

    prev_T   = T.copy()
    prev_psi = psi.copy()

    for _it in range(solver_cfg.max_outer_iters):
        u, w = _velocities(psi, dx, dz)

        # adaptive time step
        u_mag = float(np.max(np.abs(u[fluid]) + np.abs(w[fluid])) + 1e-6)
        dt = min(
            0.4 * min(dx, dz) / u_mag,
            0.45 * min(dx**2, dz**2) / max(2.0 * alpha, 1e-12),
            float(solver_cfg.max_inner_T_adv),
        )

        # ── temperature update ─────────────────────────────────────────────
        adv_T  = _upwind_adv(T, u, w, dx, dz)
        lap_T  = _laplacian(T, dx, dz)
        T_trial = T + dt * (-adv_T + alpha * lap_T)
        rT = float(solver_cfg.relax_T)
        T = (1.0 - rT) * T + rT * T_trial

        # ── temperature boundary conditions ────────────────────────────────
        T[0, :]  = cfg.T_facade_hot   # hot south facade
        T[-1, :] = cfg.T_facade_cool  # cool north facade
        T[:, 0]  = cfg.T_floor        # floor
        T[:, -1] = cfg.T_ceiling      # ceiling
        # window inflow overrides the facade BC at the opening
        if cfg.window_open > 0.0:
            T[0,  win_s] = cfg.T_outdoor  # inflow air temperature at south window
            T[-1, win_n] = T[-2, win_n]   # outflow: zero gradient at north window
        # partition cells: no Dirichlet override → diffusion makes them adiabatic

        # ── vorticity update ───────────────────────────────────────────────
        dTdx = _grad_x(T, dx)
        adv_o = _upwind_adv(omega, u, w, dx, dz)
        lap_o = _laplacian(omega, dx, dz)
        dt_o = min(
            0.35 * min(dx, dz) / u_mag,
            0.45 * min(dx**2, dz**2) / max(2.0 * nu, 1e-12),
            float(solver_cfg.max_inner_T_adv),
        )
        omega_trial = omega + dt_o * (-adv_o + nu * lap_o + cfg.g * cfg.beta * dTdx)
        r_o = float(solver_cfg.relax_omega)
        omega = (1.0 - r_o) * omega + r_o * omega_trial
        omega[solid] = 0.0

        # ── streamfunction Poisson (Jacobi sweeps) ─────────────────────────
        for _sw in range(solver_cfg.poisson_sweeps):
            pn = psi.copy()
            pn[solid] = 0.0
            lap_nbr = (pn[2:, 1:-1] + pn[:-2, 1:-1]) / dx**2 + (
                pn[1:-1, 2:] + pn[1:-1, :-2]
            ) / dz**2
            psi_int = (lap_nbr + omega[1:-1, 1:-1]) * inv_den_psi
            psi_new = pn.copy()
            psi_new[1:-1, 1:-1] = np.where(fluid[1:-1, 1:-1], psi_int, 0.0)
            # wall BCs
            psi_new[0, :]  = psi_left_bc          # left wall (0 if closed)
            psi_new[-1, :] = psi_new[-2, :]       # right wall: outflow
            psi_new[:, 0]  = 0.0                  # floor
            psi_new[:, -1] = psi_top_bc            # ceiling (0 if closed, Q if open)
            psi_new[solid] = 0.0
            psi = (1.0 - r_p) * psi + r_p * psi_new

        # ── convergence check ──────────────────────────────────────────────
        err_t = float(
            np.linalg.norm((T - prev_T)[fluid]) / (np.linalg.norm(T[fluid]) + 1e-9)
        )
        err_p = float(
            np.linalg.norm((psi - prev_psi)[fluid]) / (np.linalg.norm(psi[fluid]) + 1e-9)
        )
        prev_T   = T.copy()
        prev_psi = psi.copy()
        if err_t < solver_cfg.tol_T and err_p < solver_cfg.tol_psi:
            break

    T_out = T.copy()
    T_out[solid] = np.nan
    elapsed = time.perf_counter() - t0

    return IndoorField(
        T=T_out,
        x_grid=x,
        z_grid=z,
        method_name="pde_indoor",
        runtime_seconds=float(elapsed),
        config=cfg,
        u=u.astype(np.float64),
        w=w.astype(np.float64),
    )
