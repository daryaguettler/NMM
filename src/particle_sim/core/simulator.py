"""numpy particle + rc timestep; optional jax-jit repulsion."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from particle_sim.config.models import NumericsConfig, SimConfig
from particle_sim.core.boundary import (
    classify_exit_edge,
    hit_on_segment,
    segment_last_inside_param,
)
from particle_sim.core.geometry import LINKS, ZONE_RECTS
from particle_sim.core.physics import (
    facade_cp,
    integrate_velocity_np,
    pairwise_repulsion_np,
    wind_drive_speed,
)
from particle_sim.io.schema import N_LINKAGES, N_ZONES, TrajectoryArrays


@dataclass(frozen=True)
class CrossingRule:
    link_idx: int
    zone_from: int
    zone_to: int
    axis: str
    pos: float
    seg_lo: float
    seg_hi: float
    kind: str


def _crossing_rules() -> list[CrossingRule]:
    out: list[CrossingRule] = []
    for k, lk in enumerate(LINKS):
        if lk.kind == "doorway":
            out.append(
                CrossingRule(
                    link_idx=k,
                    zone_from=lk.zone_a,
                    zone_to=lk.zone_b,
                    axis=lk.seg_axis,
                    pos=float(lk.seg_pos),
                    seg_lo=float(lk.seg_min),
                    seg_hi=float(lk.seg_max),
                    kind=lk.kind,
                )
            )
            out.append(
                CrossingRule(
                    link_idx=k,
                    zone_from=lk.zone_b,
                    zone_to=lk.zone_a,
                    axis=lk.seg_axis,
                    pos=float(lk.seg_pos),
                    seg_lo=float(lk.seg_min),
                    seg_hi=float(lk.seg_max),
                    kind=lk.kind,
                )
            )
        else:
            out.append(
                CrossingRule(
                    link_idx=k,
                    zone_from=lk.zone_a,
                    zone_to=lk.zone_b,
                    axis=lk.seg_axis,
                    pos=float(lk.seg_pos),
                    seg_lo=float(lk.seg_min),
                    seg_hi=float(lk.seg_max),
                    kind=lk.kind,
                )
            )
    return out


RULES: list[CrossingRule] = _crossing_rules()


def _edge_code_for_rule(z_from: int, rule: CrossingRule) -> int:
    y0 = float(ZONE_RECTS[z_from, 1])
    y1 = float(ZONE_RECTS[z_from, 3])
    x0 = float(ZONE_RECTS[z_from, 0])
    x1 = float(ZONE_RECTS[z_from, 2])
    if rule.axis == "y":
        return 2 if abs(rule.pos - y0) < 0.02 else 3 if abs(rule.pos - y1) < 0.02 else (
            2 if abs(rule.pos - y0) < abs(rule.pos - y1) else 3
        )
    return 0 if abs(rule.pos - x0) < 0.02 else 1


def _passage(kind: str, nu: NumericsConfig) -> float:
    if kind == "window":
        return float(nu.passage_window)
    if kind == "doorway":
        return float(nu.passage_doorway)
    return float(nu.passage_crack)


def _opening_for_link(link_idx: int, opening6: np.ndarray) -> float:
    floor = link_idx // 5
    r = link_idx % 5
    if r == 0:
        return float(opening6[2 * floor])
    if r == 1:
        return float(opening6[2 * floor + 1])
    return 1.0


def _leeward_weights(wind_from_deg: float, wind_speed: float, opening6: np.ndarray) -> np.ndarray:
    w = np.zeros(N_LINKAGES, dtype=np.float64)
    for k, lk in enumerate(LINKS):
        if lk.zone_b >= 0:
            continue
        az = 180.0 if lk.facade == "front" else 0.0
        cp = facade_cp(wind_from_deg, az)
        if cp >= 0:
            continue
        u = _opening_for_link(k, opening6)
        if lk.kind == "crack":
            u = 1.0
        w[k] = abs(cp) * u * max(float(wind_speed), 0.05)
    if w.sum() <= 1e-12:
        w = np.array(
            [1.0 if lk.zone_b < 0 else 0.0 for lk in LINKS],
            dtype=np.float64,
        )
    return w


def initial_state(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nu = cfg.numerics
    rects = ZONE_RECTS
    n = N_ZONES * nu.n_particles_per_zone
    pos = np.zeros((n, 2), dtype=np.float64)
    z = np.repeat(np.arange(N_ZONES, dtype=np.int32), nu.n_particles_per_zone)
    for zi in range(N_ZONES):
        m = z == zi
        nn = int(m.sum())
        x0, y0, x1, y1 = rects[zi]
        pos[m, 0] = rng.uniform(x0 + 0.08, x1 - 0.08, size=nn)
        pos[m, 1] = rng.uniform(y0 + 0.08, y1 - 0.08, size=nn)
    vel = rng.normal(0.0, 0.05, size=(n, 2))
    tp = np.full(n, 22.0, dtype=np.float64)
    return pos, vel, tp, z


def _gather_rect(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = ZONE_RECTS[z]
    return r[:, 0], r[:, 1], r[:, 2], r[:, 3]


def _reflect_vel(edge: int, vx: float, vy: float) -> tuple[float, float]:
    if edge == 0:
        return abs(vx), vy
    if edge == 1:
        return -abs(vx), vy
    if edge == 2:
        return vx, abs(vy)
    if edge == 3:
        return vx, -abs(vy)
    return vx, vy


def _nudge_inside(hit: np.ndarray, edge: int, xmin: float, ymin: float, xmax: float, ymax: float, eps: float) -> np.ndarray:
    x, y = float(hit[0]), float(hit[1])
    if edge == 0:
        x = xmin + eps
    elif edge == 1:
        x = xmax - eps
    elif edge == 2:
        y = ymin + eps
    elif edge == 3:
        y = ymax - eps
    return np.array([x, y], dtype=np.float64)


def _spawn_from_link(rng: np.random.Generator, link_idx: int, t_out: float) -> tuple[np.ndarray, int, float]:
    lk = LINKS[link_idx]
    zin = lk.zone_a
    x0, y0, x1, y1 = ZONE_RECTS[zin]
    eps = 0.06
    if lk.seg_axis == "y":
        x = float(rng.uniform(lk.seg_min + 0.02, lk.seg_max - 0.02))
        if abs(lk.seg_pos - y0) < 0.02:
            y = y0 + eps
        else:
            y = y1 - eps
        p = np.array([x, y], dtype=np.float64)
    else:
        y = float(rng.uniform(lk.seg_min, lk.seg_max))
        x = x0 + eps if abs(lk.seg_pos - x0) < 0.02 else x1 - eps
        p = np.array([x, y], dtype=np.float64)
    return p, zin, float(t_out)


def simulate_trajectory(
    cfg: SimConfig,
    T_out: np.ndarray,
    wind_speed: np.ndarray,
    wind_dir: np.ndarray,
    Q_sol: np.ndarray,
    Q_int: np.ndarray,
    openings: np.ndarray,
    seed: int = 0,
    *,
    use_jit_forces: bool = False,
) -> TrajectoryArrays:
    rng = np.random.default_rng(seed)
    nu = cfg.numerics
    phy = cfg.physics
    geom = cfg.geometry
    substeps = nu.resolved_substeps()
    dt = float(nu.dt_sim)
    rho = nu.rho_air
    cp_air = float(nu.cp_air)
    vol = float(geom.long_m * geom.half_depth_m * geom.ceiling_m)
    m_particle = rho * vol / float(nu.n_particles_per_zone)
    aw = 2.0 * (geom.long_m + geom.half_depth_m) * float(geom.ceiling_m)
    aw += geom.long_m * geom.half_depth_m
    a_part = aw / float(nu.n_particles_per_zone)

    pos, vel, tp, zone = initial_state(cfg, rng)

    if use_jit_forces:
        r_cut = float(nu.r_cut)
        k_rep = float(nu.k_rep)

        @jax.jit
        def rep_jax(p: jnp.ndarray, zc: jnp.ndarray) -> jnp.ndarray:
            diff = p[:, None, :] - p[None, :, :]
            dist = jnp.linalg.norm(diff, axis=-1)
            dist = jnp.maximum(dist, 0.12 * r_cut) + 1e-9
            nloc = p.shape[0]
            ii = jnp.arange(nloc)
            mask = (zc[:, None] == zc[None, :]) & (dist < r_cut) & (ii[:, None] != ii[None, :])
            mag = k_rep * jnp.exp(-dist / r_cut) / dist
            return jnp.sum(mask[..., None] * mag[..., None] * diff, axis=1)

        def repulsion(p_loc: np.ndarray, z_loc: np.ndarray) -> np.ndarray:
            return np.asarray(rep_jax(jnp.asarray(p_loc), jnp.asarray(z_loc)))

    else:

        def repulsion(p_loc: np.ndarray, z_loc: np.ndarray) -> np.ndarray:
            return pairwise_repulsion_np(p_loc, z_loc, nu.r_cut, nu.k_rep)

    tw = np.full(N_ZONES, float(T_out[0]), dtype=np.float64)
    tm = np.full(N_ZONES, float(T_out[0]), dtype=np.float64)
    n_out = int(T_out.shape[0])
    dt_out = float(nu.dt_output)
    traj_t = np.arange(n_out, dtype=np.float64) * dt_out
    tz_traj = np.zeros((n_out, N_ZONES), dtype=np.float64)
    tw_traj = np.zeros((n_out, N_ZONES), dtype=np.float64)
    tm_traj = np.zeros((n_out, N_ZONES), dtype=np.float64)
    flow_traj = np.zeros((n_out, N_LINKAGES), dtype=np.float64)

    n_parts = pos.shape[0]

    for j in range(n_out):
        t_out = float(T_out[j])
        ws = float(wind_speed[j])
        wd = float(wind_dir[j])
        qsol = Q_sol[j].astype(np.float64)
        qint = Q_int[j].astype(np.float64)
        op6 = openings[j].astype(np.float64)
        flow_bucket = np.zeros(N_LINKAGES, dtype=np.float64)

        for _ in range(substeps):
            xmin, ymin, xmax, ymax = _gather_rect(zone)
            cnt = np.bincount(zone, minlength=N_ZONES)
            tsum = np.bincount(zone, weights=tp, minlength=N_ZONES)
            t_mean = tsum / np.maximum(cnt, 1)

            f = repulsion(pos, zone)
            ang = rng.uniform(0.0, 2.0 * np.pi, size=n_parts)
            dtz = tp - t_mean[zone]
            f = f + (nu.buoyancy_strength * np.abs(dtz))[:, None] * np.stack(
                [np.cos(ang), np.sin(ang)], axis=1
            )

            gain_w = 0.35
            for zi in range(N_ZONES):
                m = zone == zi
                if not np.any(m):
                    continue
                x0, y0, x1, y1 = ZONE_RECTS[zi]
                is_front = (zi % 2) == 0
                if is_front:
                    cp_f = facade_cp(wd, 180.0)
                    vd = wind_drive_speed(cp_f, ws)
                    lyr = m & (pos[:, 1] < y0 + nu.forcing_layer_depth)
                    uo = float(op6[zi])
                    f[lyr, 1] += gain_w * uo * vd
                else:
                    cp_f = facade_cp(wd, 0.0)
                    vd = wind_drive_speed(cp_f, ws)
                    lyr = m & (pos[:, 1] > y1 - nu.forcing_layer_depth)
                    uo = float(op6[zi])
                    f[lyr, 1] -= gain_w * uo * vd

            vel = integrate_velocity_np(vel, f, dt, float(nu.damping_coeff), float(nu.v_max))
            p0 = pos.copy()
            p_try = pos + vel * dt
            xmin, ymin, xmax, ymax = _gather_rect(zone)
            ins = (
                (p_try[:, 0] >= xmin)
                & (p_try[:, 0] <= xmax)
                & (p_try[:, 1] >= ymin)
                & (p_try[:, 1] <= ymax)
            )
            pos = p_try.copy()
            need = np.nonzero(~ins)[0]

            q_wall_zone = np.zeros(N_ZONES, dtype=np.float64)

            for i in need:
                zf = int(zone[i])
                x0, y0, x1, y1 = ZONE_RECTS[zf]
                hit = segment_last_inside_param(
                    p0[i : i + 1],
                    p_try[i : i + 1],
                    np.array([x0]),
                    np.array([y0]),
                    np.array([x1]),
                    np.array([y1]),
                )[1]
                hitp = hit[0]
                edge = int(
                    classify_exit_edge(
                        hit[0:1], np.array([x0]), np.array([y0]), np.array([x1]), np.array([y1])
                    )[0]
                )
                vx, vy = float(vel[i, 0]), float(vel[i, 1])
                passed = False
                for rule in RULES:
                    if rule.zone_from != zf:
                        continue
                    ec = _edge_code_for_rule(zf, rule)
                    if not hit_on_segment(hitp, edge, rule.axis, rule.pos, rule.seg_lo, rule.seg_hi, ec):
                        continue
                    uo = _opening_for_link(rule.link_idx, op6)
                    if rule.kind == "crack":
                        uo = 1.0
                    p_pass = _passage(rule.kind, nu)
                    if rng.random() > float(uo * p_pass):
                        continue
                    passed = True
                    m_cross = m_particle
                    if rule.zone_to < 0:
                        flow_bucket[rule.link_idx] += m_cross
                        wv = _leeward_weights(wd, ws, op6)
                        wv = np.where(
                            np.array([lk.zone_b < 0 for lk in LINKS], dtype=np.float64),
                            wv,
                            0.0,
                        )
                        if wv.sum() <= 1e-12:
                            wv = np.array(
                                [1.0 if lk.zone_b < 0 else 0.0 for lk in LINKS],
                                dtype=np.float64,
                            )
                        pk = int(rng.choice(N_LINKAGES, p=wv / wv.sum()))
                        pos[i], zone[i], tp[i] = _spawn_from_link(rng, pk, t_out)
                        vel[i] = rng.normal(0.0, 0.2, size=2)
                        flow_bucket[pk] -= m_cross
                    else:
                        zt = int(rule.zone_to)
                        if rule.kind == "doorway":
                            fsgn = 1.0 if zt > zf else -1.0
                            flow_bucket[rule.link_idx] += fsgn * m_cross
                        else:
                            flow_bucket[rule.link_idx] += m_cross
                        zone[i] = zt
                        pos[i, 0] = float(rng.uniform(rule.seg_lo + 0.02, rule.seg_hi - 0.02))
                        yn0, yn1 = float(ZONE_RECTS[zt, 1]), float(ZONE_RECTS[zt, 3])
                        vel[i] *= 0.5
                    break

                if not passed:
                    nvx, nvy = _reflect_vel(edge, vx, vy)
                    vel[i, 0], vel[i, 1] = nvx, nvy
                    pos[i] = _nudge_inside(hitp, edge, x0, y0, x1, y1, 0.04)

            xmin, ymin, xmax, ymax = _gather_rect(zone)
            px, py = pos[:, 0], pos[:, 1]
            dw = np.minimum(
                np.minimum(px - xmin, xmax - px),
                np.minimum(py - ymin, ymax - py),
            )
            near = dw < float(nu.thermal_layer_depth)
            q_part = float(phy.h_int) * a_part * (tw[zone] - tp)
            q_flux = np.where(near, q_part, 0.0)
            tp = tp + (q_flux * dt) / (m_particle * cp_air)
            for zi in range(N_ZONES):
                q_wall_zone[zi] = np.sum(q_flux[zone == zi])
            pin = qint[zone] * dt / (m_particle * cp_air * float(nu.n_particles_per_zone))
            tp = tp + pin

            tw = tw + (dt / phy.c_wall_j_per_k) * (
                phy.ua_out_w_per_k * (t_out - tw) - q_wall_zone + qsol
            )
            cnt2 = np.bincount(zone, minlength=N_ZONES)
            tsum2 = np.bincount(zone, weights=tp, minlength=N_ZONES)
            t_air = tsum2 / np.maximum(cnt2, 1)
            tm = tm + (dt / phy.c_mass_j_per_k) * (phy.h_a_mass_w_per_k * (t_air - tm))

        cntf = np.bincount(zone, minlength=N_ZONES)
        tsumf = np.bincount(zone, weights=tp, minlength=N_ZONES)
        tz_traj[j] = tsumf / np.maximum(cntf, 1)
        tw_traj[j] = tw.copy()
        tm_traj[j] = tm.copy()
        flow_traj[j] = flow_bucket / dt_out

    return TrajectoryArrays(
        t=traj_t,
        T_zones=tz_traj,
        T_wall=tw_traj,
        T_mass=tm_traj,
        flows=flow_traj,
        T_out=np.asarray(T_out, dtype=np.float64),
        wind_speed=np.asarray(wind_speed, dtype=np.float64),
        wind_dir=np.asarray(wind_dir, dtype=np.float64),
        Q_sol=np.asarray(Q_sol, dtype=np.float64),
        Q_int=np.asarray(Q_int, dtype=np.float64),
        openings=np.asarray(openings, dtype=np.float64),
    )
