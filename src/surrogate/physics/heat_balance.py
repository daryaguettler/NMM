"""explicit multi-node heat balance with inter-zone enthalpy exchange."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def heat_step(
    T_air: jnp.ndarray,
    T_wall: jnp.ndarray,
    T_mass: jnp.ndarray,
    mdot_links: jnp.ndarray,
    zone_a: jnp.ndarray,
    zone_b: jnp.ndarray,
    q_sol: jnp.ndarray,
    q_int: jnp.ndarray,
    t_out: jnp.ndarray,
    dt: float,
    c_air: float,
    c_wall: float,
    c_mass: float,
    hA_wall: float,
    hA_mass: float,
    ua_out: float,
    cp_air: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One output step update; mdot positive a->b carries enthalpy at T_air[a]."""

    def ent_step(k, carry):
        ia = zone_a[k]
        ib = zone_b[k]
        mk = mdot_links[k]
        q_a = -mk * cp_air * T_air[ia]
        q_b = mk * cp_air * T_air[ia]
        carry2 = carry.at[ia].add(q_a)
        return jnp.where(ib >= 0, carry2.at[ib].add(q_b), carry2)

    q_adv0 = jnp.zeros_like(T_air)
    q_adv = jax.lax.fori_loop(0, mdot_links.shape[0], ent_step, q_adv0)

    q_wall_net = hA_wall * (T_air - T_wall)
    q_mass_net = hA_mass * (T_air - T_mass)
    t_wall = T_wall + (dt / c_wall) * (-q_wall_net + q_sol + ua_out * (t_out - T_wall))
    t_mass = T_mass + (dt / c_mass) * (-q_mass_net)
    t_air = T_air + (dt / c_air) * (q_adv - q_wall_net - q_mass_net + q_int + ua_out * (t_out - T_air))
    return t_air, t_wall, t_mass


def physics_caps(physics) -> tuple[float, float, float, float, float, float, float]:
    """Unpack surrogate PhysicsConfig capacitances/conductances."""
    return (
        float(physics.C_air_per_zone),
        float(physics.C_wall_per_zone),
        float(physics.C_mass_per_zone),
        float(physics.hA_wall),
        float(physics.hA_mass),
        float(physics.UA_outside_per_zone),
        float(physics.cp_air),
    )
