"""semi-empirical linkage mass flows from pressure differences."""

from __future__ import annotations

import jax.numpy as jnp


def _coeff_for_kind(kind: str, Cw: float, nw: float, Cd: float, nd: float, Cc: float, nc: float) -> tuple[float, float]:
    if kind == "window":
        return float(Cw), float(nw)
    if kind == "doorway":
        return float(Cd), float(nd)
    return float(Cc), float(nc)


def mass_flow_kgs(
    dp_pa: jnp.ndarray,
    opening01: jnp.ndarray,
    kind: str,
    *,
    C_window: float,
    n_window: float,
    C_doorway: float,
    n_doorway: float,
    C_crack: float,
    n_crack: float,
    rho: float,
    eps: float = 1e-4,
    m_max: float = 50.0,
) -> jnp.ndarray:
    """m_dot ~ C * u^opening * sqrt(|dp|) * rho (orifice-style)."""
    c0, n0 = _coeff_for_kind(kind, C_window, n_window, C_doorway, n_doorway, C_crack, n_crack)
    u = jnp.clip(opening01, 0.0, 1.0)
    if kind == "crack":
        u = jnp.ones_like(u)
    mag = jnp.sqrt(jnp.abs(dp_pa) + eps)
    scale = c0 * (u ** n0) * rho
    m = scale * mag * jnp.sign(dp_pa)
    return jnp.clip(m, -m_max, m_max)
