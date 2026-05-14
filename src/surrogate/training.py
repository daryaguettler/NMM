"""training loop, checkpoints, TrainingResult."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from surrogate.corpus import (
    compute_flow_scale_rms,
    initial_state_from_trajectory,
    load_particle_corpus_dir,
    load_run_trajectory,
    trajectory_to_forcings,
)
from surrogate.loss import trajectory_supervision_loss
from surrogate.surrogate import init_residual_mlp, simulate
from surrogate.types import SurrogateParams, TrainingConfig, TrainingResult


def _tx(cfg: TrainingConfig):
    o = cfg.optimizer
    parts: list = []
    if cfg.grad_clip is not None:
        parts.append(optax.clip_by_global_norm(float(cfg.grad_clip)))
    if o.name == "adam":
        parts.append(optax.adam(o.learning_rate))
    elif o.name == "adamw":
        parts.append(optax.adamw(o.learning_rate, weight_decay=o.weight_decay))
    else:
        parts.append(optax.sgd(o.learning_rate))
    return optax.chain(*parts) if len(parts) > 1 else parts[0]


def train(config: TrainingConfig) -> TrainingResult:
    if not config.surrogate.use_learned_residual:
        msg = "training requires surrogate.use_learned_residual=True"
        raise ValueError(msg)
    t0 = time.perf_counter()
    root = Path(config.corpus_path)
    manifest = load_particle_corpus_dir(root)
    train_m, val_m = manifest.split(config.val_fraction, config.val_split_seed)
    scales = jnp.asarray(compute_flow_scale_rms(train_m), dtype=jnp.float32)
    topo = train_m.topology
    phy = train_m.physics
    nz = topo.n_zones

    rng = jax.random.PRNGKey(int(config.seed))
    rng, k0 = jax.random.split(rng)
    params, _ = init_residual_mlp(
        k0,
        4 * nz + 3,
        nz,
        config.surrogate.residual_hidden_sizes,
        str(config.surrogate.residual_activation),
    )
    tx = _tx(config)
    opt_state = tx.init(params)

    def batch_loss(mlp_p, paths: list[str]) -> jnp.ndarray:
        tot = jnp.array(0.0, dtype=jnp.float32)
        for pth in paths:
            tr = load_run_trajectory(pth)
            frc = trajectory_to_forcings(tr)
            st0 = initial_state_from_trajectory(tr)
            gt_t = jnp.asarray(tr.T_air, dtype=jnp.float32)
            gt_f = jnp.asarray(tr.flows, dtype=jnp.float32)
            tout = jnp.asarray(frc.T_out, dtype=jnp.float32)
            sp = SurrogateParams(
                residual_mlp_params=mlp_p,
                residual_mlp_activation=config.surrogate.residual_activation,
            )
            pred_t, _, _, pred_f = simulate(sp, frc, st0, topo, phy, config.surrogate)
            l_a, _ = trajectory_supervision_loss(
                pred_t,
                pred_f,
                gt_t,
                gt_f,
                tout,
                topo,
                config.weights,
                scales,
            )
            tot = tot + l_a
        return tot / float(max(len(paths), 1))

    value_and_grad = jax.value_and_grad(batch_loss)
    train_paths = [r.path for r in train_m.runs]
    val_paths = [r.path for r in val_m.runs]
    n = len(train_paths)
    bs = max(1, min(int(config.batch_size), n))

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_metrics_by_epoch: list[dict[str, float]] = []
    last_rmse_t = 0.0
    last_rmse_f = 0.0

    for epoch in range(int(config.n_epochs)):
        perm = np.random.default_rng(int(config.seed) + epoch).permutation(n)
        eloss = 0.0
        steps = 0
        for i in range(0, n, bs):
            batch = [train_paths[int(j)] for j in perm[i : i + bs]]
            loss, g = value_and_grad(params, batch)
            updates, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, updates)
            eloss += float(loss)
            steps += 1
        train_losses.append(eloss / max(steps, 1))

        if epoch % int(config.val_every_n_epochs) == 0:
            eval_paths = val_paths if val_paths else train_paths[:1]
            vb = eval_paths[: min(len(eval_paths), bs)]
            vl = float(batch_loss(params, vb))
            val_losses.append(vl)
            trv = load_run_trajectory(eval_paths[0])
            frcv = trajectory_to_forcings(trv)
            stv = initial_state_from_trajectory(trv)
            gt_t = jnp.asarray(trv.T_air, dtype=jnp.float32)
            gt_f = jnp.asarray(trv.flows, dtype=jnp.float32)
            sp = SurrogateParams(
                residual_mlp_params=params,
                residual_mlp_activation=config.surrogate.residual_activation,
            )
            pt, _, _, pf = simulate(sp, frcv, stv, topo, phy, config.surrogate)
            last_rmse_t = float(jnp.sqrt(jnp.mean((pt - gt_t) ** 2)))
            last_rmse_f = float(jnp.sqrt(jnp.mean((pf - gt_f) ** 2)))
            val_metrics_by_epoch.append(
                {"epoch": float(epoch), "val_loss": vl, "rmse_T": last_rmse_t, "rmse_flow": last_rmse_f}
            )

        if (epoch + 1) % int(config.checkpoint_every_n_epochs) == 0:
            ck_dir = Path(config.output_dir)
            ck_dir.mkdir(parents=True, exist_ok=True)
            ck_path = ck_dir / f"epoch_{epoch + 1:05d}.pkl"
            with ck_path.open("wb") as f:
                pickle.dump(jax.tree.map(lambda x: np.asarray(x), params), f)

    return TrainingResult(
        final_params=SurrogateParams(
            residual_mlp_params=params,
            residual_mlp_activation=config.surrogate.residual_activation,
        ),
        train_losses=np.asarray(train_losses, dtype=np.float64),
        val_losses=np.asarray(val_losses, dtype=np.float64),
        val_metrics_by_epoch=val_metrics_by_epoch,
        config=config,
        runtime_seconds=float(time.perf_counter() - t0),
        final_val_rmse_temperature=last_rmse_t,
        final_val_rmse_flow=last_rmse_f,
    )
