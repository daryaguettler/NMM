"""joint scenario sample for corpus rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from particle_sim.config.models import SimConfig
from particle_sim.scenarios.policies import PolicyName, policy_schedule
from particle_sim.scenarios.weather import (
    boston_summer_hours,
    hourly_to_output,
    solar_and_internal,
)


@dataclass
class ScenarioBundle:
    T_out_h: np.ndarray
    wind_speed_h: np.ndarray
    wind_dir_h: np.ndarray
    Q_sol_h: np.ndarray
    Q_int_h: np.ndarray
    openings_h: np.ndarray
    policy: PolicyName | str
    meta: dict[str, Any]


def sample_scenario(
    cfg: SimConfig,
    rng: np.random.Generator,
    *,
    policy: PolicyName | None = None,
) -> ScenarioBundle:
    nh = int(np.ceil(float(cfg.corpus.duration_hours)))
    T_out, ws, wd = boston_summer_hours(nh, rng)
    is_front = np.array([i % 2 == 0 for i in range(6)], dtype=np.float64)
    qsol, qint = solar_and_internal(nh, rng, is_front_mask=is_front)
    policies: list[PolicyName] = [
        "always_closed",
        "always_open",
        "daytime",
        "night_flush",
        "random",
        "threshold",
    ]
    if policy is None:
        policy = str(rng.choice(np.array(policies)))
    tzone_proxy = np.tile(T_out[:, None], (1, 6)) + rng.normal(0.0, 0.5, size=(nh, 6))
    op = policy_schedule(
        policy,  # type: ignore[arg-type]
        nh,
        rng,
        T_zone_hourly=tzone_proxy,
        T_out_hourly=T_out,
    )
    if rng.random() < 0.5:
        for zi in range(6):
            if rng.random() < 0.2:
                subpol = str(rng.choice(np.array(policies)))
                col = policy_schedule(
                    subpol,  # type: ignore[arg-type]
                    nh,
                    rng,
                    T_zone_hourly=tzone_proxy[:, zi : zi + 1],
                    T_out_hourly=T_out,
                )[:, 0]
                op[:, zi] = col
    return ScenarioBundle(
        T_out_h=T_out,
        wind_speed_h=ws,
        wind_dir_h=wd,
        Q_sol_h=qsol,
        Q_int_h=qint,
        openings_h=op,
        policy=policy,
        meta={"duration_hours": nh},
    )


def resample_for_sim(
    bundle: ScenarioBundle,
    sim: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nu = sim.numerics
    n_out = int(np.floor(sim.corpus.duration_hours * 3600.0 / nu.dt_output))
    dt_out = float(nu.dt_output)
    return (
        hourly_to_output(bundle.T_out_h, n_out, dt_out),
        hourly_to_output(bundle.wind_speed_h, n_out, dt_out),
        hourly_to_output(bundle.wind_dir_h, n_out, dt_out),
        hourly_to_output(bundle.Q_sol_h, n_out, dt_out),
        hourly_to_output(bundle.Q_int_h, n_out, dt_out),
        hourly_to_output(bundle.openings_h, n_out, dt_out),
    )
