"""nodal pressure solve for quasi-steady airflows each output step."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from surrogate.physics.cp import facade_cp
from surrogate.physics.flow_law import mass_flow_kgs
from surrogate.types import Topology


@dataclass(frozen=True)
class PressureAux:
    zone_a: jnp.ndarray  # (K,) int32
    zone_b: jnp.ndarray  # (K,) int32, -1 outdoor
    kind_codes: jnp.ndarray  # (K,) int32 0 win 1 door 2 crack
    facade_az: jnp.ndarray  # (K,) float32


def topology_to_pressure_aux(topology: Topology) -> PressureAux:
    za = []
    zb = []
    kinds = []
    faz = []
    for lk in topology.linkages:
        za.append(topology.zone_index(lk.a))
        if lk.b == "outdoor":
            zb.append(-1)
        else:
            zb.append(topology.zone_index(lk.b))
        kinds.append(0 if lk.kind == "window" else 1 if lk.kind == "doorway" else 2)
        az = 180.0 if lk.facade_azimuth is None else float(lk.facade_azimuth)
        faz.append(az)
    return PressureAux(
        zone_a=jnp.asarray(za, dtype=jnp.int32),
        zone_b=jnp.asarray(zb, dtype=jnp.int32),
        kind_codes=jnp.asarray(kinds, dtype=jnp.int32),
        facade_az=jnp.asarray(faz, dtype=jnp.float32),
    )


jax.tree_util.register_pytree_node(
    PressureAux,
    lambda a: ((a.zone_a, a.zone_b, a.kind_codes, a.facade_az), None),
    lambda _aux, children: PressureAux(*children),
)


def _link_mass_vector(
    dp: jnp.ndarray,
    op: jnp.ndarray,
    kind_codes: jnp.ndarray,
    *,
    Cw: float,
    nw: float,
    Cd: float,
    nd: float,
    Cc: float,
    nc: float,
    rho: float,
) -> jnp.ndarray:
    m_win = mass_flow_kgs(
        dp,
        op,
        "window",
        C_window=Cw,
        n_window=nw,
        C_doorway=Cd,
        n_doorway=nd,
        C_crack=Cc,
        n_crack=nc,
        rho=rho,
    )
    m_door = mass_flow_kgs(
        dp,
        op,
        "doorway",
        C_window=Cw,
        n_window=nw,
        C_doorway=Cd,
        n_doorway=nd,
        C_crack=Cc,
        n_crack=nc,
        rho=rho,
    )
    m_crack = mass_flow_kgs(
        dp,
        jnp.ones_like(op),
        "crack",
        C_window=Cw,
        n_window=nw,
        C_doorway=Cd,
        n_doorway=nd,
        C_crack=Cc,
        n_crack=nc,
        rho=rho,
    )
    return jnp.where(
        kind_codes == 0,
        m_win,
        jnp.where(kind_codes == 1, m_door, m_crack),
    )


def _scatter_residual(mdot: jnp.ndarray, aux: PressureAux, n_zones: int) -> jnp.ndarray:
    def step(k, r):
        ia = aux.zone_a[k]
        ib = aux.zone_b[k]
        mk = mdot[k]
        r2 = r.at[ia].add(-mk)
        return jnp.where(ib >= 0, r2.at[ib].add(mk), r2)

    r0 = jnp.zeros((n_zones,), dtype=jnp.float32)
    return jax.lax.fori_loop(0, mdot.shape[0], step, r0)


@partial(jax.jit, static_argnames=("n_zones", "max_iter", "rho", "Cp_amp", "Cw", "nw", "Cd", "nd", "Cc", "nc", "tol"))
def newton_pressures(
    pr0: jnp.ndarray,
    aux: PressureAux,
    openings6: jnp.ndarray,
    wind_speed: jnp.ndarray,
    wind_dir_met: jnp.ndarray,
    *,
    n_zones: int,
    rho: float,
    Cp_amp: float,
    Cw: float,
    nw: float,
    Cd: float,
    nd: float,
    Cc: float,
    nc: float,
    max_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    ia = aux.zone_a
    ib = aux.zone_b
    kc = aux.kind_codes
    wind_from = jnp.mod(wind_dir_met + 180.0, 360.0)
    cpv = facade_cp(wind_from, aux.facade_az)
    u_eff = jnp.sign(cpv) * jnp.sqrt(jnp.abs(cpv) + 1e-6) * wind_speed
    head_k = 0.5 * rho * (Cp_amp * u_eff) ** 2

    def full_p(pr: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([jnp.zeros((1,), dtype=pr.dtype), pr], axis=0)

    def flows_from_p(P: jnp.ndarray) -> jnp.ndarray:
        p_b = jnp.where(ib >= 0, P[ib], -head_k)
        dp = P[ia] - p_b
        op = jnp.where(ib < 0, openings6[ia], 0.5 * (openings6[ia] + openings6[jnp.maximum(ib, 0)]))
        op = jnp.where(kc == 2, jnp.ones_like(op), op)
        return _link_mass_vector(dp, op, kc, Cw=Cw, nw=nw, Cd=Cd, nd=nd, Cc=Cc, nc=nc, rho=rho)

    def residual(pr: jnp.ndarray) -> jnp.ndarray:
        P = full_p(pr)
        md = flows_from_p(P)
        R = _scatter_residual(md, aux, n_zones)
        return R[1:]

    def one(i, pr):
        R = residual(pr)
        J = jax.jacfwd(residual)(pr)
        step = jnp.linalg.solve(J + 1e-4 * jnp.eye(n_zones - 1, dtype=J.dtype), R)
        pr2 = pr - step
        return jnp.where(jnp.linalg.norm(R) < tol, pr, pr2)

    pr = jax.lax.fori_loop(0, max_iter, one, pr0)
    P = full_p(pr)
    md = flows_from_p(P)
    return P, md
