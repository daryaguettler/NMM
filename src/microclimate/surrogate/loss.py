"""supervised mse + optional advection-diffusion residual on collocation points."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from microclimate.surrogate.model import Params, forward, pack_inputs


def mse_loss(
    params: Params,
    inp: jax.Array,
    target: jax.Array,
    activation: str,
) -> jax.Array:
    pred = jax.vmap(lambda row: forward(params, row, activation))(inp)
    return jnp.mean((pred - target) ** 2)


def _T_point_fn(
    params: Params,
    weather: jax.Array,
    activation: str,
    x_scale: float,
    z_scale: float,
    u_scale: float,
    t_scale: float,
    t0: float,
) -> Callable[[jax.Array], jax.Array]:
    def T_at(xz: jax.Array) -> jax.Array:
        row = pack_inputs(
            jnp.asarray([xz[0]], dtype=jnp.float32),
            jnp.asarray([xz[1]], dtype=jnp.float32),
            weather,
            x_scale,
            z_scale,
            u_scale,
            t_scale,
            t0,
        )[0]
        return forward(params, row, activation)

    return T_at


def physics_residual_loss(
    params: Params,
    xz: jax.Array,
    u_w: jax.Array,
    weather: jax.Array,
    alpha: float | jax.Array,
    activation: str,
    x_scale: float,
    z_scale: float,
    u_scale: float,
    t_scale: float,
    t0: float,
) -> jax.Array:
    """Xz (n,2), u_w (n,2) with columns u,w at same points; penalizes (u.Tx + w.Tz - alpha lap T)^2."""
    a = jnp.asarray(alpha, dtype=jnp.float32)
    T_at = _T_point_fn(
        params, weather, activation, x_scale, z_scale, u_scale, t_scale, t0
    )
    g = jax.grad(T_at)
    H = jax.hessian(T_at)

    def one_res(pt: jax.Array, uw: jax.Array) -> jax.Array:
        d = g(pt)
        h = H(pt)
        lap = h[0, 0] + h[1, 1]
        adv = uw[0] * d[0] + uw[1] * d[1]
        return (adv - a * lap) ** 2

    r = jax.vmap(one_res)(xz, u_w)
    return jnp.mean(r)


def combined_loss(
    params: Params,
    inp_sup: jax.Array,
    tgt_sup: jax.Array,
    xz_col: jax.Array | None,
    uw_col: jax.Array | None,
    weather: jax.Array,
    lam_phys: float,
    alpha: float | jax.Array,
    activation: str,
    x_scale: float,
    z_scale: float,
    u_scale: float,
    t_scale: float,
    t0: float,
) -> jax.Array:
    L = mse_loss(params, inp_sup, tgt_sup, activation)
    if lam_phys > 0.0 and xz_col is not None and uw_col is not None and xz_col.shape[0] > 0:
        L = L + lam_phys * physics_residual_loss(
            params,
            xz_col,
            uw_col,
            weather,
            alpha,
            activation,
            x_scale,
            z_scale,
            u_scale,
            t_scale,
            t0,
        )
    return L
