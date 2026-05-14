"""opening schedules for windows (6 knobs) per hour."""

from __future__ import annotations

from typing import Literal

import numpy as np

PolicyName = Literal[
    "always_closed",
    "always_open",
    "daytime",
    "night_flush",
    "random",
    "threshold",
]


def policy_schedule(
    name: PolicyName,
    n_hours: int,
    rng: np.random.Generator,
    *,
    T_zone_hourly: np.ndarray | None = None,
    T_out_hourly: np.ndarray | None = None,
) -> np.ndarray:
    """Shape (n_hours, 6) opening factors."""
    h = np.arange(n_hours)
    hours = h % 24
    if name == "always_closed":
        return np.zeros((n_hours, 6), dtype=np.float64)
    if name == "always_open":
        return np.ones((n_hours, 6), dtype=np.float64)
    if name == "daytime":
        m = (hours >= 8) & (hours <= 20)
        v = np.where(m, 1.0, 0.0)
        return np.tile(v[:, None], (1, 6))
    if name == "night_flush":
        m = (hours >= 22) | (hours <= 6)
        v = np.where(m, 1.0, 0.0)
        return np.tile(v[:, None], (1, 6))
    if name == "random":
        out = np.zeros((n_hours, 6), dtype=np.float64)
        sw = int(rng.integers(1, 7))
        cur = rng.random(size=6)
        block = 0
        for i in range(n_hours):
            if i % sw == 0:
                cur = rng.random(size=6)
            out[i] = cur
        return out
    if name == "threshold":
        if T_zone_hourly is None or T_out_hourly is None:
            return np.zeros((n_hours, 6), dtype=np.float64)
        open_m = (T_zone_hourly > 26.0) & (
            T_out_hourly[:, None] < (T_zone_hourly - 2.0)
        )
        return open_m.astype(np.float64)
    return np.zeros((n_hours, 6), dtype=np.float64)
