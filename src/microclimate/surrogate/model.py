"""shallow mlp for t(x,z, weather) in jax."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

ParamLayer = tuple[jax.Array, jax.Array]  # W, b
Params = list[ParamLayer]


def _act(x: jax.Array, name: Literal["gelu", "relu", "tanh"]) -> jax.Array:
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "relu":
        return jax.nn.relu(x)
    return jnp.tanh(x)


def init_mlp(
    rng: jax.Array,
    layer_sizes: list[int],
    scale: float = 0.08,
    activation: Literal["gelu", "relu", "tanh"] = "gelu",
) -> Params:
    """layer_sizes: in, hidden..., out."""
    params: Params = []
    keys = jax.random.split(rng, max(1, len(layer_sizes) - 1))
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        w = jax.random.normal(keys[i], (n_out, n_in), dtype=jnp.float32) * scale
        b = jnp.zeros((n_out,), dtype=jnp.float32)
        params.append((w, b))
    _ = activation  # reserved for symmetry with call sites
    return params


def forward(params: Params, inp: jax.Array, activation: str) -> jax.Array:
    """Inp shape (in_dim,); returns scalar temperature."""
    h = inp
    for i, (w, b) in enumerate(params):
        z = w @ h + b
        h = _act(z, activation) if i < len(params) - 1 else z  # type: ignore[arg-type]
    return h.squeeze()


def forward_batch(
    params: Params, inp: jax.Array, activation: str
) -> jax.Array:
    """Inp shape (batch, in_dim); returns (batch,) temperatures."""
    return jax.vmap(lambda row: forward(params, row, activation))(inp)


def pack_inputs(
    x: jax.Array,
    z: jax.Array,
    weather: jax.Array,
    x_scale: float,
    z_scale: float,
    u_scale: float,
    t_scale: float,
    t0: float,
) -> jax.Array:
    """x,z shape (batch,); weather (3,) = U_ref, T_facade, T_ref."""
    w0 = jnp.full_like(x, weather[0], dtype=jnp.float32)
    w1 = jnp.full_like(x, weather[1], dtype=jnp.float32)
    w2 = jnp.full_like(x, weather[2], dtype=jnp.float32)
    return jnp.stack(
        [
            x / x_scale,
            z / z_scale,
            (w0 - 2.0) / u_scale,
            (w1 - t0) / t_scale,
            (w2 - t0) / t_scale,
        ],
        axis=-1,
    )
