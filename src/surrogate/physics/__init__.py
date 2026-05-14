"""jax physics kernels for surrogate."""

from surrogate.physics.cp import facade_cp, wind_dynamic_head_pa
from surrogate.physics.flow_law import mass_flow_kgs
from surrogate.physics.heat_balance import heat_step, physics_caps
from surrogate.physics.pressure_solver import newton_pressures, topology_to_pressure_aux

__all__ = [
    "facade_cp",
    "heat_step",
    "mass_flow_kgs",
    "newton_pressures",
    "physics_caps",
    "topology_to_pressure_aux",
    "wind_dynamic_head_pa",
]
