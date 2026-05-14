"""kernels for forces, wind cp, damping."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def clamp_magnitude_vec(v: np.ndarray, v_max: float) -> np.ndarray:
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.where(mag > v_max, v_max / (mag + 1e-12), 1.0)
    return v * scale


def jnp_clamp_magnitude_vec(v: jnp.ndarray, v_max: float) -> jnp.ndarray:
    mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
    scale = jnp.where(mag > v_max, v_max / (mag + 1e-12), 1.0)
    return v * scale


def pairwise_repulsion_np(
    pos: np.ndarray,
    zone: np.ndarray,
    r_cut: float,
    k_rep: float,
) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    raw = np.linalg.norm(diff, axis=-1)
    dist = np.maximum(raw, 0.12 * r_cut) + 1e-9
    n = pos.shape[0]
    ii = np.arange(n)
    mask = (
        (zone[:, None] == zone[None, :])
        & (dist < r_cut)
        & (ii[:, None] != ii[None, :])
    )
    mag = k_rep * np.exp(-dist / r_cut) / dist
    return np.sum(mask[..., None] * mag[..., None] * diff, axis=1)


def facade_cp(wind_from_deg: float, facade_azimuth_deg: float) -> float:
    a = np.deg2rad(wind_from_deg - facade_azimuth_deg)
    return float(np.clip(np.cos(a), -1.0, 1.0))


def wind_drive_speed(cp: float, wind_speed: float) -> float:
    return float(np.sign(cp) * np.sqrt(abs(cp) + 1e-9) * wind_speed)


def integrate_velocity_np(
    v: np.ndarray,
    f: np.ndarray,
    dt: float,
    damping: float,
    v_max: float,
) -> np.ndarray:
    v = v + f * dt
    v = v * float(np.exp(-damping * dt))
    return clamp_magnitude_vec(v, v_max)
