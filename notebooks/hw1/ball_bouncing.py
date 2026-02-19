import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from pydantic import BaseModel, Field, computed_field, ConfigDict

class Ball(BaseModel):
    """a 3d ball with material and state"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    radius: float
    density: float
    color: str = "blue"
    elasticity: float
    max_deformation: float = 0.1
    stiffness: float = 500.0
    dilation: float = 0.0

    start_position: list[float]
    start_velocity: list[float]
    current_position: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=float))
    current_velocity: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=float))
    angular_velocity: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=float))
    acceleration: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=float))
    force: np.ndarray = Field(default_factory=lambda: np.zeros(3, dtype=float))

    @computed_field
    @property
    def mass(self) -> float:
        return (4.0 / 3.0) * np.pi * (self.radius ** 3) * self.density

    @computed_field
    @property
    def moment_of_inertia(self) -> float:
        return (2.0 / 5.0) * self.mass * (self.radius ** 2)

    def initialize_state(self) -> None:
        self.current_position = np.array(self.start_position, dtype=float)
        self.current_velocity = np.array(self.start_velocity, dtype=float)
        self.angular_velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
        self.dilation = 0.0


medium_types = Literal["air", "helium", "nitrogen"]
class BounceEnvironment(BaseModel):
    """environment parameters for the room"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    room_dimensions: list[float]
    gravity: float
    gravity_vector: list[float] = [0.0, 0.0, -1.0]
    linear_drag: float = 0.0
    fluid_density: float = 1.225
    drag_coefficient: float = 0.47
    medium_type: medium_types = "air"
    wall_restitution: float = 0.9
    wall_curvature: float = 0.0


class Simulation(BaseModel):
    """run a physics simulation for multiple balls"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    balls: list[Ball]
    environment: BounceEnvironment
    time_step: float
    total_time_steps: int
    positions: list[np.ndarray] = Field(default_factory=list)
    velocities: list[np.ndarray] = Field(default_factory=list)
    occupancy_grid: np.ndarray | None = None
    occupancy_history: list[np.ndarray] = Field(default_factory=list)

    def _gravity_vector(self) -> np.ndarray:
        g_dir = np.array(self.environment.gravity_vector, dtype=float)
        g_norm = np.linalg.norm(g_dir)
        if g_norm == 0.0:
            return np.zeros(3, dtype=float)
        return (g_dir / g_norm) * self.environment.gravity

    def calculate_forces(self) -> None:
        g_vec = self._gravity_vector()
        env = self.environment
        for ball in self.balls:
            ball.force = ball.mass * g_vec
            v = ball.current_velocity
            speed = np.linalg.norm(v)
            if env.linear_drag != 0.0:
                ball.force += -env.linear_drag * v
            if speed > 0.0 and env.fluid_density > 0.0:
                area = np.pi * (ball.radius ** 2)
                quad_mag = 0.5 * env.fluid_density * env.drag_coefficient * area * speed
                ball.force += -quad_mag * v

        count = len(self.balls)
        for i in range(count):
            for j in range(i + 1, count):
                b1, b2 = self.balls[i], self.balls[j]
                delta = b2.current_position - b1.current_position
                dist = np.linalg.norm(delta)
                radius_sum = b1.radius + b2.radius
                if dist >= radius_sum or dist <= 0.0:
                    continue
                normal = delta / dist
                overlap = radius_sum - dist
                k_eff = min(b1.stiffness, b2.stiffness)
                f_mag = k_eff * overlap
                self.balls[i].force += f_mag * normal
                self.balls[j].force -= f_mag * normal

    def update_acceleration(self) -> None:
        for ball in self.balls:
            ball.acceleration = ball.force / ball.mass

    def update_velocity(self) -> None:
        for ball in self.balls:
            ball.current_velocity = (
                ball.current_velocity + ball.acceleration * self.time_step
            )

    def update_position(self) -> None:
        for ball in self.balls:
            ball.current_position = (
                ball.current_position + ball.current_velocity * self.time_step
            )

    def update_deformation(self) -> None:
        recovery_rate = 2.0
        for ball in self.balls:
            ball.dilation = max(0.0, ball.dilation - recovery_rate * self.time_step)

    def bounce_off_walls(self, ball: Ball) -> bool:
        dims = np.array(self.environment.room_dimensions, dtype=float)
        pos = ball.current_position.copy()
        vel = ball.current_velocity.copy()
        omega = ball.angular_velocity.copy()
        r = ball.radius
        e_eff = ball.elasticity * self.environment.wall_restitution
        hit = False
        wall_friction = 0.2
        friction_applied = False

        for axis in range(3):
            min_bound = r
            max_bound = dims[axis] - r
            e_axis = np.zeros(3, dtype=float)
            e_axis[axis] = 1.0

            colliding_low = pos[axis] < min_bound and vel[axis] < 0
            colliding_high = pos[axis] > max_bound and vel[axis] > 0

            if colliding_low or colliding_high:
                # save incoming velocity for dilation and friction before modifying
                incoming_speed = abs(vel[axis])
                r_vec = -r * e_axis if colliding_low else r * e_axis

                # compute friction using incoming velocity
                if not friction_applied and ball.mass > 0 and ball.moment_of_inertia > 0:
                    v_cp = vel + np.cross(omega, r_vec)
                    v_tang = v_cp - np.dot(v_cp, e_axis) * e_axis
                    v_tang_mag = np.linalg.norm(v_tang)
                    if v_tang_mag > 1e-10:
                        j_t = -wall_friction * v_tang
                        vel += j_t / ball.mass
                        omega += np.cross(r_vec, j_t) / ball.moment_of_inertia
                        friction_applied = True

                # apply normal impulse (bounce)
                pos[axis] = min_bound if colliding_low else max_bound
                vel[axis] = -vel[axis] * e_eff
                ball.dilation = min(ball.max_deformation, ball.dilation + incoming_speed * 0.01)
                hit = True

        ball.current_position = pos
        ball.current_velocity = vel
        ball.angular_velocity = omega
        return hit

    def handle_ball_collisions(self) -> bool:
        count = len(self.balls)
        had_collision = False

        for i in range(count):
            for j in range(i + 1, count):
                b1 = self.balls[i]
                b2 = self.balls[j]
                delta = b2.current_position - b1.current_position
                dist = np.linalg.norm(delta)
                min_dist = b1.radius + b2.radius

                if dist >= min_dist:
                    continue

                had_collision = True

                if dist == 0.0:
                    normal = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    normal = delta / dist

                overlap = min_dist - dist
                inv_m1 = 1.0 / b1.mass if b1.mass > 0.0 else 0.0
                inv_m2 = 1.0 / b2.mass if b2.mass > 0.0 else 0.0
                total_inv_mass = inv_m1 + inv_m2
                if total_inv_mass > 0.0:
                    b1.current_position -= normal * (overlap * (inv_m1 / total_inv_mass))
                    b2.current_position += normal * (overlap * (inv_m2 / total_inv_mass))

                rel_vel = b2.current_velocity - b1.current_velocity
                vel_along_normal = np.dot(rel_vel, normal)
                if vel_along_normal > 0.0:
                    continue

                # dilation based on impact velocity, not overlap
                impact_speed = abs(vel_along_normal)
                b1.dilation = min(b1.max_deformation, b1.dilation + impact_speed * 0.01)
                b2.dilation = min(b2.max_deformation, b2.dilation + impact_speed * 0.01)

                e_eff = np.sqrt(b1.elasticity * b2.elasticity)
                impulse_mag = (-(1.0 + e_eff) * vel_along_normal) / total_inv_mass
                impulse = impulse_mag * normal

                b1.current_velocity -= impulse * inv_m1
                b2.current_velocity += impulse * inv_m2

        return had_collision

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.stack([b.current_position for b in self.balls], axis=0)
        velocities = np.stack([b.current_velocity for b in self.balls], axis=0)
        return positions, velocities

    def init_grid(self, resolution: int = 20) -> None:
        self.occupancy_grid = np.zeros((resolution, resolution, resolution), dtype=float)

    def update_occupancy(self) -> None:
        if self.occupancy_grid is None:
            return
        dims = np.array(self.environment.room_dimensions, dtype=float)
        res = self.occupancy_grid.shape[0]
        for ball in self.balls:
            idx = ((ball.current_position / dims) * (res - 1)).astype(int)
            idx = np.clip(idx, 0, res - 1)
            self.occupancy_grid[tuple(idx)] += 1.0

    def clamp_to_bounds(self) -> None:
        """position-only clamp after ball-ball separation"""
        dims = np.array(self.environment.room_dimensions, dtype=float)
        for ball in self.balls:
            r = ball.radius
            ball.current_position = np.clip(
                ball.current_position, r, dims - r,
            )

    def step(self, store: bool = True) -> None:
        self.calculate_forces()
        self.update_acceleration()
        self.update_velocity()
        self.update_position()
        self.update_deformation()
        for ball in self.balls:
            self.bounce_off_walls(ball)
        self.handle_ball_collisions()
        self.clamp_to_bounds()
        if store:
            positions, velocities = self._snapshot()
            self.positions.append(positions)
            self.velocities.append(velocities)
        self.update_occupancy()
        if store and self.occupancy_grid is not None:
            self.occupancy_history.append(self.occupancy_grid.sum(axis=2).copy())

    def simulate(self) -> None:
        for ball in self.balls:
            ball.initialize_state()
        positions, velocities = self._snapshot()
        self.positions = [positions]
        self.velocities = [velocities]
        self.init_grid()
        self.occupancy_history = []
        self.update_occupancy()
        if self.occupancy_grid is not None:
            self.occupancy_history.append(self.occupancy_grid.sum(axis=2).copy())
        for _ in range(self.total_time_steps):
            self.step()

    def simulate_until(self, stop_event, max_steps: int = 500_000, store_every: int = 10, max_frames: int = 3000) -> None:
        for ball in self.balls:
            ball.initialize_state()
        positions, velocities = self._snapshot()
        self.positions = [positions]
        self.velocities = [velocities]
        self.init_grid()
        self.occupancy_history = []
        self.update_occupancy()
        if self.occupancy_grid is not None:
            self.occupancy_history.append(self.occupancy_grid.sum(axis=2).copy())
        for i in range(max_steps):
            if stop_event.is_set():
                break
            store = (i % store_every == 0) and len(self.positions) < max_frames
            self.step(store=store)