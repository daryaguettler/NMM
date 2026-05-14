"""option A (trajectory) + option B (gradient supervision) losses."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surrogate.types import LossWeights, Topology


def linkage_weight_vector(topology: Topology, w: LossWeights) -> jnp.ndarray:
    out = []
    for lk in topology.linkages:
        if lk.kind == "window":
            out.append(float(w.flow_window))
        elif lk.kind == "doorway":
            out.append(float(w.flow_doorway))
        else:
            out.append(float(w.flow_crack))
    return jnp.asarray(out, dtype=jnp.float32)


def heatwave_time_weights(t_out: jnp.ndarray, w: LossWeights) -> jnp.ndarray:
    m = (t_out > float(w.heatwave_weight_T_threshold)).astype(jnp.float32)
    return 1.0 + (float(w.heatwave_weight_multiplier) - 1.0) * m


def trajectory_supervision_loss(
    pred_T: jnp.ndarray,
    pred_flows: jnp.ndarray,
    gt_T: jnp.ndarray,
    gt_flows: jnp.ndarray,
    t_out: jnp.ndarray,
    topology: Topology,
    w: LossWeights,
    flow_scale: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Option A: mse temperature + weighted normalized flow mse."""
    lk_w = linkage_weight_vector(topology, w)
    tw = heatwave_time_weights(t_out, w)
    l_t = jnp.mean(tw[:, None] * (pred_T - gt_T) ** 2)
    denom = jnp.maximum(flow_scale, 1e-6)
    err_f = (pred_flows - gt_flows) / denom
    wf = tw[:, None] * (err_f**2) * lk_w[None, :]
    l_f = jnp.mean(wf) if w.normalize_flows_per_linkage else jnp.mean((pred_flows - gt_flows) ** 2)
    wall = jnp.array(0.0)
    mass = jnp.array(0.0)
    total = (
        float(w.temperature) * l_t
        + float(w.flows) * l_f
        + float(w.wall_temp) * wall
        + float(w.mass_temp) * mass
    )
    return total, {"temp": l_t, "flow": l_f}


def finite_difference_gradient(
    fn,
    x0: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    g = np.zeros_like(x0)
    for i in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        g.flat[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return g


def gradient_supervision_loss(
    grad_surrogate: jnp.ndarray,
    grad_target: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    if weight <= 0.0:
        return jnp.array(0.0, dtype=jnp.float32)
    return float(weight) * jnp.mean((grad_surrogate - grad_target) ** 2)
