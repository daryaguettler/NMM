"""wind pressure coefficients (facade cp) in jax."""

from __future__ import annotations

import jax.numpy as jnp


def facade_cp(
    wind_from_deg: float | jnp.ndarray,
    facade_azimuth_deg: float | jnp.ndarray,
) -> jnp.ndarray:
    wf = jnp.asarray(wind_from_deg, dtype=jnp.float32)
    fa = jnp.asarray(facade_azimuth_deg, dtype=jnp.float32)
    a = jnp.deg2rad(wf - fa)
    return jnp.clip(jnp.cos(a), -1.0, 1.0)


def wind_dynamic_head_pa(
    wind_speed: jnp.ndarray,
    wind_from_deg: jnp.ndarray,
    facade_azimuth_deg: float,
    rho: float,
    cp_scale: float,
) -> jnp.ndarray:
    """0.5 * rho * U_eff^2 with U_eff from signed cp."""
    cpv = facade_cp(wind_from_deg, facade_azimuth_deg)
    u_eff = jnp.sign(cpv) * jnp.sqrt(jnp.abs(cpv) + 1e-6) * wind_speed
    return 0.5 * rho * (cp_scale * u_eff) ** 2
