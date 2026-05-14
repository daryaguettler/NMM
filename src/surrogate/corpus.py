"""corpus adapters: particle_sim manifest/npz -> typed surrogate objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import particle_sim.io.schema as ps_io
from surrogate.topology_builder import build_default_topology
from surrogate.types import (
    CorpusManifest,
    CorpusRun,
    Forcings,
    PhysicsConfig,
    State,
    Trajectory,
)


def weather_seed_from_window(weather_window: str) -> int:
    s = str(weather_window)
    if "synthetic_seed=" in s:
        return int(s.split("=", maxsplit=1)[1])
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:4], "big", signed=False) % (2**31)


def default_physics_from_particle_json(cfg_path: Path) -> PhysicsConfig:
    if not cfg_path.exists():
        return PhysicsConfig()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    phy = data.get("physics") or {}
    num = data.get("numerics") or {}
    return PhysicsConfig(
        rho_air=float(num.get("rho_air", 1.2)),
        cp_air=float(num.get("cp_air", 1005.0)),
        C_wall_per_zone=float(phy.get("c_wall_j_per_k", 4.8e6)),
        C_mass_per_zone=float(phy.get("c_mass_j_per_k", 4.8e6)),
        UA_outside_per_zone=float(phy.get("ua_out_w_per_k", 16.0)),
        hA_wall=float(phy.get("h_int", 185.0)) * 24.0,
        hA_mass=float(phy.get("h_a_mass_w_per_k", 246.0)),
        dt_output=float(num.get("dt_output", 600.0)),
    )


def load_particle_corpus_dir(root: Path | str, *, physics: PhysicsConfig | None = None) -> CorpusManifest:
    base = Path(root)
    m = ps_io.manifest_from_json(base / "manifest.json")
    phy = physics if physics is not None else default_physics_from_particle_json(base / "config.json")
    topo = build_default_topology()
    runs: list[CorpusRun] = []
    for item in m.runs:
        runs.append(
            CorpusRun(
                run_id=int(item.run_id),
                scenario_hash=str(item.scenario_hash),
                path=str(base / "runs" / f"run_{int(item.run_id):06d}.npz"),
                n_timesteps=int(item.n_timesteps),
                weather_seed=weather_seed_from_window(str(item.weather_window)),
                source="particle_sim",
            )
        )
    return CorpusManifest(
        name=base.name,
        topology=topo,
        physics=phy,
        n_runs=len(runs),
        runs=runs,
    )


def trajectory_arrays_to_surrogate(traj: ps_io.TrajectoryArrays) -> Trajectory:
    return Trajectory(
        t=traj.t,
        T_air=traj.T_zones,
        T_wall=traj.T_wall,
        T_mass=traj.T_mass,
        flows=traj.flows,
        T_out=traj.T_out,
        wind_speed=traj.wind_speed,
        wind_dir=traj.wind_dir,
        openings=traj.openings,
        Q_sol=traj.Q_sol,
        Q_int=traj.Q_int,
    )


def load_run_trajectory(path: Path | str) -> Trajectory:
    return Trajectory.from_npz(path)


def trajectory_to_forcings(traj: Trajectory, dt: float | None = None) -> Forcings:
    d = float(dt) if dt is not None else float(np.asarray(traj.t)[1] - np.asarray(traj.t)[0])
    qs = traj.Q_sol if traj.Q_sol is not None else np.zeros((traj.n_steps, 6))
    qi = traj.Q_int if traj.Q_int is not None else np.zeros((traj.n_steps, 6))
    return Forcings(
        T_out=traj.T_out,
        wind_speed=traj.wind_speed,
        wind_dir=traj.wind_dir,
        Q_sol=qs,
        Q_int=qi,
        openings=traj.openings,
        dt=d,
    )


def initial_state_from_trajectory(traj: Trajectory) -> State:
    ta = np.asarray(traj.T_air)
    tw = np.asarray(traj.T_wall)
    tm = np.asarray(traj.T_mass)
    return State(T_air=ta[0], T_wall=tw[0], T_mass=tm[0])


def load_sim_config(cfg_path: Path | str):
    """particle_sim SimConfig from corpus config.json (strips extra keys)."""
    from particle_sim.config.models import SimConfig

    raw = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    payload = {k: raw[k] for k in ("geometry", "numerics", "physics", "corpus") if k in raw}
    return SimConfig.model_validate(payload)


def compute_flow_scale_rms(manifest: CorpusManifest) -> np.ndarray:
    """per-linkage rms flow magnitude on training manifest runs."""
    acc = None
    n_runs = 0.0
    for r in manifest.runs:
        tr = load_run_trajectory(r.path)
        f = np.asarray(tr.flows)
        if acc is None:
            acc = np.zeros((f.shape[1],), dtype=np.float64)
        acc += np.mean(f**2, axis=0)
        n_runs += 1.0
    if acc is None:
        return np.ones((15,), dtype=np.float64)
    return np.sqrt(np.maximum(acc / np.maximum(n_runs, 1.0), 1e-12))
