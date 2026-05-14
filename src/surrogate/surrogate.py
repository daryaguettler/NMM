"""jax surrogate forward: pressure + heat balance (+ optional residual mlp)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from surrogate.physics.heat_balance import heat_step, physics_caps
from surrogate.physics.pressure_solver import newton_pressures, topology_to_pressure_aux
from surrogate.types import (
    Forcings,
    PhysicsConfig,
    State,
    SurrogateConfig,
    SurrogateParams,
    Topology,
)


def _act_fn(name: str, x: jnp.ndarray) -> jnp.ndarray:
    if name == "relu":
        return jnp.maximum(x, 0.0)
    if name == "tanh":
        return jnp.tanh(x)
    return jax.nn.gelu(x, approximate=True)


def init_residual_mlp(
    rng: jax.Array,
    in_dim: int,
    out_dim: int,
    hidden: tuple[int, ...],
    act: str,
) -> tuple[dict[str, Any], str]:
    keys = jax.random.split(rng, len(hidden) + 1)
    wlist = []
    blist = []
    d0 = in_dim
    for i, h in enumerate(hidden):
        k1, k2 = jax.random.split(keys[i])
        w = 0.02 * jax.random.normal(k1, (d0, h))
        b = 0.02 * jax.random.normal(k2, (h,))
        wlist.append(w)
        blist.append(b)
        d0 = h
    kf = keys[-1]
    w = 0.02 * jax.random.normal(kf, (d0, out_dim))
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    params = {"Ws": wlist, "bs": blist, "Wout": w, "bout": b}
    return params, act


def apply_residual_mlp(p: dict[str, Any], x: jnp.ndarray, act: str) -> jnp.ndarray:
    h = x
    for w, b in zip(p["Ws"], p["bs"], strict=True):
        h = _act_fn(act, h @ w + b)
    return h @ p["Wout"] + p["bout"]


def simulate(
    params: SurrogateParams,
    forcings: Forcings,
    initial_state: State,
    topology: Topology,
    physics: PhysicsConfig,
    cfg: SurrogateConfig,
    *,
    openings_override: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (T_air, T_wall, T_mass, flows) each (T,nz) or (T,K) for flows."""
    aux = topology_to_pressure_aux(topology)
    T_out = jnp.asarray(forcings.T_out, dtype=jnp.float32)
    ws = jnp.asarray(forcings.wind_speed, dtype=jnp.float32)
    wd = jnp.asarray(forcings.wind_dir, dtype=jnp.float32)
    qsol = jnp.asarray(forcings.Q_sol, dtype=jnp.float32)
    qint = jnp.asarray(forcings.Q_int, dtype=jnp.float32)
    op = (
        jnp.asarray(openings_override, dtype=jnp.float32)
        if openings_override is not None
        else jnp.asarray(forcings.openings, dtype=jnp.float32)
    )

    t_air0 = jnp.asarray(initial_state.T_air, dtype=jnp.float32).ravel()
    t_wall0 = jnp.asarray(initial_state.T_wall, dtype=jnp.float32).ravel()
    t_mass0 = jnp.asarray(initial_state.T_mass, dtype=jnp.float32).ravel()
    nz = topology.n_zones
    dt = float(forcings.dt)
    caps = physics_caps(physics)
    c_air, c_wall, c_mass, hAw, hAm, ua0, cp_air = caps

    def scan_fn(carry, j):
        ta, tw, tm = carry
        toe = T_out[j]
        wj = ws[j]
        dj = wd[j]
        qs = qsol[j]
        qi = qint[j]
        oj = op[j]
        pr_init = jnp.zeros((nz - 1), dtype=jnp.float32)
        if cfg.use_pressure_solver:
            _, md = newton_pressures(
                pr_init,
                aux,
                oj,
                wj,
                dj,
                n_zones=nz,
                rho=float(physics.rho_air),
                Cp_amp=float(physics.Cp_amplitude),
                Cw=float(physics.C_window),
                nw=float(physics.n_window),
                Cd=float(physics.C_doorway),
                nd=float(physics.n_doorway),
                Cc=float(physics.C_crack),
                nc=float(physics.n_crack),
                max_iter=int(cfg.newton_max_iter),
                tol=float(cfg.newton_tol),
            )
        else:
            md = jnp.zeros((topology.n_linkages,), dtype=jnp.float32)
        if cfg.use_heat_balance:
            ta, tw, tm = heat_step(
                ta,
                tw,
                tm,
                md,
                aux.zone_a,
                aux.zone_b,
                qs,
                qi,
                toe,
                dt,
                c_air,
                c_wall,
                c_mass,
                hAw,
                hAm,
                ua0,
                cp_air,
            )
        if cfg.use_learned_residual and params.residual_mlp_params is not None:
            feat = jnp.concatenate([ta, tw, tm, oj, jnp.asarray([toe, wj, dj], dtype=jnp.float32)])
            mlp_p = params.residual_mlp_params
            if mlp_p is None:
                msg = "residual_mlp_params required when use_learned_residual"
                raise RuntimeError(msg)
            ta = ta + apply_residual_mlp(mlp_p, feat, params.residual_mlp_activation)
        return (ta, tw, tm), jnp.concatenate([ta, tw, tm, md], axis=0)

    n = int(T_out.shape[0])
    carry0 = (t_air0, t_wall0, t_mass0)
    _, stacked = lax.scan(scan_fn, carry0, jnp.arange(n, dtype=jnp.int32))
    ta_tr = stacked[:, :nz]
    tw_tr = stacked[:, nz : 2 * nz]
    tm_tr = stacked[:, 2 * nz : 3 * nz]
    fl_tr = stacked[:, 3 * nz :]
    return ta_tr, tw_tr, tm_tr, fl_tr
