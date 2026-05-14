"""validation harnesses: pressure network, coupled surrogate, inverse design."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from particle_sim.core.simulator import simulate_trajectory
from surrogate.corpus import (
    initial_state_from_trajectory,
    load_run_trajectory,
    load_sim_config,
    trajectory_to_forcings,
)
from surrogate.physics.pressure_solver import newton_pressures, topology_to_pressure_aux
from surrogate.surrogate import simulate
from surrogate.types import (
    CoupledValidation,
    InverseDesignValidation,
    PhysicsConfig,
    PressureSolverValidation,
    SurrogateConfig,
    SurrogateParams,
    Topology,
)


def validate_pressure_solver(
    traj_path: str | Path,
    topology: Topology,
    physics: PhysicsConfig,
    *,
    step_index: int = 1,
    sur: SurrogateConfig,
) -> PressureSolverValidation:
    traj = load_run_trajectory(traj_path)
    j = int(step_index)
    aux = topology_to_pressure_aux(topology)
    nz = topology.n_zones
    oj = jnp.asarray(traj.openings[j], dtype=jnp.float32)
    ws = jnp.asarray(traj.wind_speed[j], dtype=jnp.float32)
    wd = jnp.asarray(traj.wind_dir[j], dtype=jnp.float32)
    pr0 = jnp.zeros((nz - 1,), jnp.float32)
    _, pred = newton_pressures(
        pr0,
        aux,
        oj,
        ws,
        wd,
        n_zones=nz,
        rho=float(physics.rho_air),
        Cp_amp=float(physics.Cp_amplitude),
        Cw=float(physics.C_window),
        nw=float(physics.n_window),
        Cd=float(physics.C_doorway),
        nd=float(physics.n_doorway),
        Cc=float(physics.C_crack),
        nc=float(physics.n_crack),
        max_iter=int(sur.newton_max_iter),
        tol=float(sur.newton_tol),
    )
    gt = np.asarray(traj.flows[j], dtype=np.float64)
    pr = np.asarray(pred, dtype=np.float64)
    err = pr - gt
    rmse = np.abs(err)
    bias = err
    r2v = np.full_like(gt, np.nan)
    return PressureSolverValidation(
        flows_true=gt,
        flows_pred=pr,
        rmse_per_linkage=rmse,
        r2_per_linkage=r2v,
        bias_per_linkage=bias,
    )


def validate_coupled(
    traj_path: str | Path,
    topology: Topology,
    physics: PhysicsConfig,
    sur: SurrogateConfig,
    params: SurrogateParams | None = None,
    *,
    threshold_c: float = 28.0,
) -> CoupledValidation:
    traj = load_run_trajectory(traj_path)
    frc = trajectory_to_forcings(traj)
    st0 = initial_state_from_trajectory(traj)
    sp = params or SurrogateParams()
    pred_t, _, _, pred_f = simulate(sp, frc, st0, topology, physics, sur)
    gt_t = np.asarray(traj.T_air, dtype=np.float64)
    gt_f = np.asarray(traj.flows, dtype=np.float64)
    pt = np.asarray(pred_t)
    pf = np.asarray(pred_f)
    rmse_z = np.sqrt(np.mean((pt - gt_t) ** 2, axis=0))
    rmse_run = np.sqrt(np.mean((pt - gt_t) ** 2))
    ta_mean_t = np.mean(gt_t, axis=1)
    pa_mean_t = np.mean(pt, axis=1)
    htrue = np.sum(ta_mean_t > threshold_c)
    hpred = np.sum(pa_mean_t > threshold_c)
    return CoupledValidation(
        T_air_true=gt_t,
        T_air_pred=pt,
        flows_true=gt_f,
        flows_pred=pf,
        rmse_T_per_zone=rmse_z,
        rmse_T_per_run=np.array([float(rmse_run)]),
        hours_above_threshold_true=np.array([float(htrue)]),
        hours_above_threshold_pred=np.array([float(hpred)]),
        hours_above_threshold_error=np.array([float(hpred - htrue)]),
    )


def validate_inverse_design(
    traj_path: str | Path,
    corpus_dir: str | Path,
    topology: Topology,
    physics: PhysicsConfig,
    sur: SurrogateConfig,
    *,
    threshold_c: float = 28.0,
    opt_steps: int = 40,
    seed: int = 0,
) -> InverseDesignValidation:
    traj = load_run_trajectory(traj_path)
    f0 = trajectory_to_forcings(traj)
    st0 = initial_state_from_trajectory(traj)
    t_steps = int(f0.n_steps)
    base = Path(corpus_dir)

    def sur_penalty(op6: jnp.ndarray) -> jnp.ndarray:
        op = jnp.broadcast_to(op6[None, :], (t_steps, 6))
        ta, _, _, _ = simulate(
            SurrogateParams(),
            f0,
            st0,
            topology,
            physics,
            sur,
            openings_override=op,
        )
        return jnp.mean(jnp.maximum(ta - float(threshold_c), 0.0))

    op = jnp.full((6,), 0.35, dtype=jnp.float32)
    tx = optax.adam(0.12)
    ost = tx.init(op)
    losses: list[float] = []
    for _ in range(int(opt_steps)):
        v, g = jax.value_and_grad(sur_penalty)(op)
        losses.append(float(v))
        u, ost = tx.update(g, ost, op)
        op = optax.apply_updates(op, u)
        op = jnp.clip(op, 0.05, 1.0)

    op_np = np.asarray(op, dtype=np.float64)
    op_full = np.broadcast_to(op_np[None, :], (t_steps, 6))
    sim_cfg = load_sim_config(base / "config.json")
    sim_tr = simulate_trajectory(
        sim_cfg,
        np.asarray(traj.T_out),
        np.asarray(traj.wind_speed),
        np.asarray(traj.wind_dir),
        np.asarray(traj.Q_sol if traj.Q_sol is not None else np.zeros((t_steps, 6))),
        np.asarray(traj.Q_int if traj.Q_int is not None else np.zeros((t_steps, 6))),
        op_full.astype(np.float64),
        seed=int(seed),
    )
    sa = float(np.sum(np.mean(sim_tr.T_zones, axis=1) > threshold_c))
    ta, _, _, _ = simulate(
        SurrogateParams(),
        f0,
        st0,
        topology,
        physics,
        sur,
        openings_override=jnp.asarray(op_full, dtype=jnp.float32),
    )
    ss = float(np.sum(np.mean(np.asarray(ta), axis=1) > threshold_c))
    return InverseDesignValidation(
        initial_openings=np.full((6,), 0.35),
        optimized_openings=op_np,
        surrogate_loss_trajectory=np.asarray(losses),
        surrogate_final_hours_above=ss,
        simulator_final_hours_above=sa,
        discrepancy=float(abs(ss - sa)),
    )
