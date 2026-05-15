"""optax training loop over corpus runs."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from microclimate.config import SurrogateConfig
from microclimate.surrogate.data import load_all_runs, run_to_flat_arrays
from microclimate.surrogate.loss import combined_loss
from microclimate.surrogate.model import init_mlp

DEFAULT_SCALES = {
    "x_scale": 60.0,
    "z_scale": 30.0,
    "u_scale": 5.0,
    "t_scale": 25.0,
    "t0": 22.0,
}


def train_surrogate(
    corpus_dir: Path,
    cfg: SurrogateConfig,
    val_fraction: float = 0.2,
    val_seed: int = 42,
    max_points_per_run: int | None = 8000,
) -> dict:
    root = Path(corpus_dir).expanduser().resolve()
    runs_root = root / "runs"
    runs_all = load_all_runs(root)
    if len(runs_all) == 0:
        msg = (
            f"no npz runs under {runs_root} (expected run_*.npz).\n"
            "use a merged microclimate corpus directory, or generate one with "
            "microclimate.apps.build_corpus_shard + cluster.merge_shards."
        )
        raise ValueError(msg)
    if len(runs_all) == 1:
        # no hold-out possible; train and val metrics use the same run
        train_runs = list(runs_all)
        val_runs = list(runs_all)
    else:
        rng_split = np.random.default_rng(val_seed)
        idx = np.arange(len(runs_all))
        rng_split.shuffle(idx)
        n_val = max(1, int(np.round(len(runs_all) * val_fraction)))
        val_idx = set(idx[:n_val].tolist())
        train_runs = [runs_all[i] for i in range(len(runs_all)) if i not in val_idx]
        val_runs = [runs_all[i] for i in range(len(runs_all)) if i in val_idx]

    rng_data = np.random.default_rng(int(cfg.seed))
    layer_sizes = [5, *cfg.hidden_sizes, 1]
    rng = jax.random.PRNGKey(int(cfg.seed))
    rng, k_init = jax.random.split(rng)
    params = init_mlp(k_init, layer_sizes, scale=0.08, activation=str(cfg.activation))
    tx = optax.adam(float(cfg.learning_rate))
    opt_state = tx.init(params)

    scales = DEFAULT_SCALES
    x_scale = scales["x_scale"]
    z_scale = scales["z_scale"]
    u_scale = scales["u_scale"]
    t_scale = scales["t_scale"]
    t0f = scales["t0"]
    act = str(cfg.activation)
    lam = float(cfg.physics_weight)
    n_coll = int(cfg.n_collocation_per_batch)

    @jax.jit
    def _step(p, os, inp, tgt, xz_c, uw_c, wj, alpha_j):
        def loss_fn(pp):
            return combined_loss(
                pp,  # type: ignore[arg-type]
                inp,
                tgt,
                xz_c,
                uw_c,
                wj,
                lam,
                alpha_j,
                act,
                x_scale,
                z_scale,
                u_scale,
                t_scale,
                t0f,
            )

        loss, grad = jax.value_and_grad(loss_fn)(p)
        updates, os2 = tx.update(grad, os, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, os2, loss

    hist: dict[str, list] = {"train_loss": [], "val_loss": [], "step": []}
    t0_wall = time.perf_counter()

    for it in range(int(cfg.n_iterations)):
        run = train_runs[it % len(train_runs)]
        xf, zf, _, tf, uf, wf = run_to_flat_arrays(run, max_points_per_run, rng_data)
        weather = jnp.asarray(
            [run.cfg.U_ref, run.cfg.T_facade_hot, run.cfg.T_ref], dtype=jnp.float32
        )
        inp_np = np.stack(
            [
                xf / x_scale,
                zf / z_scale,
                np.full_like(xf, run.cfg.U_ref - 2.0) / u_scale,
                np.full_like(xf, run.cfg.T_facade_hot - t0f) / t_scale,
                np.full_like(xf, run.cfg.T_ref - t0f) / t_scale,
            ],
            axis=1,
        ).astype(np.float32)
        tgt_np = tf.astype(np.float32)
        inp = jnp.asarray(inp_np)
        tgt = jnp.asarray(tgt_np)

        if n_coll > 0 and uf is not None and wf is not None and lam > 0.0:
            pick = rng_data.choice(xf.size, size=min(n_coll, xf.size), replace=False)
            xz_c = jnp.asarray(
                np.stack([xf[pick], zf[pick]], axis=1), dtype=jnp.float32
            )
            uw_c = jnp.asarray(
                np.stack([uf[pick], wf[pick]], axis=1), dtype=jnp.float32
            )
        else:
            xz_c = jnp.zeros((0, 2), dtype=jnp.float32)
            uw_c = jnp.zeros((0, 2), dtype=jnp.float32)

        alpha_j = jnp.asarray(float(run.cfg.alpha), dtype=jnp.float32)
        params, opt_state, loss_t = _step(
            params, opt_state, inp, tgt, xz_c, uw_c, weather, alpha_j
        )

        if it % 50 == 0 or it == int(cfg.n_iterations) - 1:
            hist["train_loss"].append(float(loss_t))
            v_acc: list[float] = []
            for vr in val_runs:
                vxf, vzf, _, vtf, vuf, vwf = run_to_flat_arrays(
                    vr, max_points_per_run, rng_data
                )
                vin = jnp.asarray(
                    np.stack(
                        [
                            vxf / x_scale,
                            vzf / z_scale,
                            np.full_like(vxf, vr.cfg.U_ref - 2.0) / u_scale,
                            np.full_like(vxf, vr.cfg.T_facade_hot - t0f) / t_scale,
                            np.full_like(vxf, vr.cfg.T_ref - t0f) / t_scale,
                        ],
                        axis=1,
                    ),
                    dtype=jnp.float32,
                )
                vt = jnp.asarray(vtf, dtype=jnp.float32)
                wj = jnp.asarray(
                    [vr.cfg.U_ref, vr.cfg.T_facade_hot, vr.cfg.T_ref], dtype=jnp.float32
                )
                if n_coll > 0 and vuf is not None and vwf is not None and lam > 0.0:
                    pk = rng_data.choice(vxf.size, size=min(n_coll, vxf.size), replace=False)
                    xz_cv = jnp.asarray(
                        np.stack([vxf[pk], vzf[pk]], axis=1), dtype=jnp.float32
                    )
                    uw_cv = jnp.asarray(
                        np.stack([vuf[pk], vwf[pk]], axis=1), dtype=jnp.float32
                    )
                else:
                    xz_cv = jnp.zeros((0, 2), dtype=jnp.float32)
                    uw_cv = jnp.zeros((0, 2), dtype=jnp.float32)
                vl = combined_loss(
                    params,
                    vin,
                    vt,
                    xz_cv,
                    uw_cv,
                    wj,
                    lam,
                    float(vr.cfg.alpha),
                    act,
                    x_scale,
                    z_scale,
                    u_scale,
                    t_scale,
                    t0f,
                )
                v_acc.append(float(vl))
            hist["val_loss"].append(float(np.mean(v_acc)))
            hist["step"].append(int(it))

    wall = time.perf_counter() - t0_wall
    return {
        "params": jax.device_get(params),
        "history": hist,
        "wall_seconds": wall,
        "train_runs": len(train_runs),
        "val_runs": len(val_runs),
        "scales": scales,
        "layer_sizes": layer_sizes,
        "activation": act,
    }


def save_artifact(out_dir: Path, bundle: dict, cfg: SurrogateConfig) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "surrogate_params.pkl").open("wb") as f:
        pickle.dump(bundle["params"], f)
    h = bundle["history"]
    meta = {
        "surrogate_config": json.loads(cfg.model_dump_json()),
        "layer_sizes": bundle["layer_sizes"],
        "activation": bundle["activation"],
        "scales": bundle["scales"],
        "history": {
            "step": h.get("step", []),
            "train_loss": h["train_loss"],
            "val_loss": h["val_loss"],
        },
        "wall_seconds": bundle["wall_seconds"],
    }
    (out_dir / "surrogate_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
