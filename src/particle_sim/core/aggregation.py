"""per-link flows -> zone net inflow (kirchhoff check)."""

from __future__ import annotations

import numpy as np

from particle_sim.core.geometry import LINKS
from particle_sim.io.schema import N_LINKAGES, N_ZONES


def zone_flow_residuals(flow_row: np.ndarray) -> np.ndarray:
    """Returns (6,) net kg/s into zone (should be ~0)."""
    if flow_row.shape != (N_LINKAGES,):
        raise ValueError(flow_row.shape)
    resid = np.zeros(N_ZONES, dtype=np.float64)
    for k, lk in enumerate(LINKS):
        f = float(flow_row[k])
        a = lk.zone_a
        b = lk.zone_b
        if b < 0:
            resid[a] -= f
        else:
            resid[a] -= f
            resid[b] += f
    return resid


def max_abs_residual_over_time(flows: np.ndarray) -> float:
    t = flows.shape[0]
    m = 0.0
    for i in range(t):
        r = np.abs(zone_flow_residuals(flows[i]))
        m = max(m, float(r.max()))
    return m


def mean_abs_residual_timeavg(flows: np.ndarray) -> float:
    acc = np.zeros(N_ZONES, dtype=np.float64)
    for i in range(flows.shape[0]):
        acc += np.abs(zone_flow_residuals(flows[i]))
    return float(acc.mean() / max(flows.shape[0], 1))
