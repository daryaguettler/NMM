"""synthetic boston summer hour series."""

from __future__ import annotations

import numpy as np


def boston_summer_hours(
    n_hours: int,
    rng: np.random.Generator,
    *,
    day_mean_range: tuple[float, float] = (18.0, 30.0),
    diurnal_amp_range: tuple[float, float] = (4.0, 12.0),
    heatwave_amp_range: tuple[float, float] = (0.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns T_out (n_hours), wind_speed, wind_dir (met deg from)."""
    mean = rng.uniform(*day_mean_range)
    amp = rng.uniform(*diurnal_amp_range)
    h = np.arange(n_hours, dtype=np.float64)
    T_out = mean + amp * np.sin(2 * np.pi * (h - 8.0) / 24.0)
    hw = rng.uniform(*heatwave_amp_range)
    T_out += hw * np.exp(-((h - n_hours / 2) / max(n_hours / 6, 1.0)) ** 2)

    mu = rng.uniform(1.5, 4.5)
    wind_speed = rng.lognormal(np.log(mu), 0.35, size=n_hours)
    wind_speed = np.clip(wind_speed, 0.2, 12.0)

    base = rng.normal(225.0, 15.0)
    wind_dir = base + 40.0 * np.sin(2 * np.pi * h / 96.0) + rng.normal(0.0, 8.0, size=n_hours)
    wind_dir = wind_dir % 360.0
    return T_out, wind_speed, wind_dir


def solar_and_internal(
    n_hours: int,
    rng: np.random.Generator,
    *,
    is_front_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """q_sol (n,6), q_int (n,6)."""
    h = np.arange(n_hours)
    hours = h % 24
    sin_solar = np.maximum(0.0, np.sin(2 * np.pi * (hours - 6) / 24.0))
    base_sol = np.outer(sin_solar, 200.0 + 500.0 * is_front_mask.astype(np.float64))
    q_sol = base_sol + rng.normal(0.0, 15.0, size=base_sol.shape)
    occ = np.where((hours < 7) | (hours > 22), 0.35, 1.0)
    q_int = np.outer(occ, np.full(6, 130.0)) + rng.normal(0.0, 10.0, size=(n_hours, 6))
    q_int = np.clip(q_int, 30.0, 400.0)
    return q_sol, q_int


def hourly_to_output(
    hourly: np.ndarray,
    n_out: int,
    dt_out: float,
) -> np.ndarray:
    sec = np.arange(n_out, dtype=np.float64) * float(dt_out)
    hi = (sec / 3600.0).astype(np.int64)
    hi = np.clip(hi, 0, hourly.shape[0] - 1)
    if hourly.ndim == 1:
        return hourly[hi].astype(np.float64)
    return hourly[hi].astype(np.float64)
