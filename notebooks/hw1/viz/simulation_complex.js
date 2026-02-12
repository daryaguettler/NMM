export class Ball {
    constructor({ radius, density, color = "blue", elasticity, startPosition, startVelocity }) {
        this.radius = radius;
        this.density = density;
        this.color = color;
        this.elasticity = elasticity;

        this.startPosition = new Float64Array(startPosition);
        this.startVelocity = new Float64Array(startVelocity);

        this.currentPosition = new Float64Array(3);
        this.currentVelocity = new Float64Array(3);

        const r3 = this.radius ** 3;
        this.mass = (4.0 / 3.0) * Math.PI * r3 * this.density;
        this.invMass = this.mass > 0 ? 1.0 / this.mass : 0;
    }

    initializeState() {
        this.currentPosition.set(this.startPosition);
        this.currentVelocity.set(this.startVelocity);
    }
}

export class BoxObstacle {
    // axis-aligned box defined by min/max corners in room coordinates
    constructor({ minCorner, maxCorner, restitution = 0.8 }) {
        this.min = minCorner.slice();
        this.max = maxCorner.slice();
        this.restitution = restitution;
    }
}

export class BounceEnvironmentComplex {
    constructor({
        roomDimensions,
        gravity,
        gravityVector = [0, 0, -1],
        wallRestitution = 0.9,
        friction = 0.1,
        sleepThreshold = 0.05,
        obstacles = [],
        tubeYRange = null,
        tubeZRange = null,
        tubeRadius = null,
        tubeCenterY = null,
        tubeCenterZ = null,
    }) {
        this.roomDimensions = roomDimensions.slice();
        this.gravity = gravity;
        this.gravityVector = gravityVector.slice();
        this.wallRestitution = wallRestitution;

        this.friction = friction;
        this.sleepThreshold = sleepThreshold;

        // simple axis-aligned box obstacles
        this.obstacles = obstacles;

        // tube that constrains entry in y and z for side-injection
        // if null, no constraint; if provided, an interval [min, max]
        this.tubeYRange = tubeYRange;
        this.tubeZRange = tubeZRange;

        // cylindrical wall parameters (visualized tube)
        this.tubeRadius = tubeRadius;
        this.tubeCenterY = tubeCenterY;
        this.tubeCenterZ = tubeCenterZ;

        const gx = this.gravityVector[0];
        const gy = this.gravityVector[1];
        const gz = this.gravityVector[2];
        const norm = Math.hypot(gx, gy, gz) || 1.0;
        const scale = this.gravity / norm;
        this.gNormalized = [
            gx * scale,
            gy * scale,
            gz * scale,
        ];
    }
}

export class SimulationComplex {
    constructor({ balls, environment, timeStep, totalTimeSteps }) {
        this.balls = balls;
        this.environment = environment;
        this.timeStep = timeStep;
        this.totalTimeSteps = totalTimeSteps;
        this.positions = [];
    }

    step(store = true) {
        const dt = this.timeStep;
        const env = this.environment;

        // 1. integration
        for (const b of this.balls) {
            b.currentVelocity[0] += env.gNormalized[0] * dt;
            b.currentVelocity[1] += env.gNormalized[1] * dt;
            b.currentVelocity[2] += env.gNormalized[2] * dt;

            b.currentPosition[0] += b.currentVelocity[0] * dt;
            b.currentPosition[1] += b.currentVelocity[1] * dt;
            b.currentPosition[2] += b.currentVelocity[2] * dt;
        }

        // 2. wall collisions
        for (const b of this.balls) {
            const wall_e = b.elasticity * env.wallRestitution;

            // always clamp in x to keep balls within tube length
            {
                const i = 0;
                const minB = b.radius;
                const maxB = env.roomDimensions[i] - b.radius;
                if (b.currentPosition[i] < minB) {
                    b.currentPosition[i] = minB;
                    if (b.currentVelocity[i] < 0) b.currentVelocity[i] *= -wall_e;
                } else if (b.currentPosition[i] > maxB) {
                    b.currentPosition[i] = maxB;
                    if (b.currentVelocity[i] > 0) b.currentVelocity[i] *= -wall_e;
                }
            }

            // if tubeRadius is defined, use cylindrical walls in y,z instead of box
            if (env.tubeRadius != null && env.tubeCenterY != null && env.tubeCenterZ != null) {
                const cy = env.tubeCenterY;
                const cz = env.tubeCenterZ;
                const dy = b.currentPosition[1] - cy;
                const dz = b.currentPosition[2] - cz;
                const r = Math.hypot(dy, dz);
                const limit = Math.max(env.tubeRadius - b.radius, 0);

                if (r > 0 && r > limit) {
                    // normal pointing inward toward tube center
                    const nx = 0;
                    const ny = dy / r;
                    const nz = dz / r;

                    // project back onto cylinder surface
                    const newR = Math.max(limit, 0);
                    const scale = newR / r;
                    b.currentPosition[1] = cy + dy * scale;
                    b.currentPosition[2] = cz + dz * scale;

                    // reflect velocity about radial normal
                    const vDotN = b.currentVelocity[1] * ny + b.currentVelocity[2] * nz;
                    if (vDotN > 0) {
                        const j = -(1 + wall_e) * vDotN;
                        b.currentVelocity[1] += ny * j;
                        b.currentVelocity[2] += nz * j;
                    }
                }
            } else {
                // fallback: original box walls for y and z
                for (let i = 1; i < 3; i++) {
                    const minB = b.radius;
                    const maxB = env.roomDimensions[i] - b.radius;
                    if (b.currentPosition[i] < minB) {
                        b.currentPosition[i] = minB;
                        if (b.currentVelocity[i] < 0) b.currentVelocity[i] *= -wall_e;
                    } else if (b.currentPosition[i] > maxB) {
                        b.currentPosition[i] = maxB;
                        if (b.currentVelocity[i] > 0) b.currentVelocity[i] *= -wall_e;
                    }
                }
            }
        }

        // 3. obstacle collisions (axis-aligned boxes)
        for (const b of this.balls) {
            for (const obs of env.obstacles) {
                // check if ball overlaps with box (expanded by ball radius)
                const minX = obs.min[0] - b.radius;
                const maxX = obs.max[0] + b.radius;
                const minY = obs.min[1] - b.radius;
                const maxY = obs.max[1] + b.radius;
                const minZ = obs.min[2] - b.radius;
                const maxZ = obs.max[2] + b.radius;

                const px = b.currentPosition[0];
                const py = b.currentPosition[1];
                const pz = b.currentPosition[2];

                if (px >= minX && px <= maxX && py >= minY && py <= maxY && pz >= minZ && pz <= maxZ) {
                    // ball overlaps with expanded box, find closest face
                    const dists = [
                        { axis: 0, dist: px - obs.min[0], normal: -1 }, // left face
                        { axis: 0, dist: obs.max[0] - px, normal: 1 },  // right face
                        { axis: 1, dist: py - obs.min[1], normal: -1 }, // bottom face
                        { axis: 1, dist: obs.max[1] - py, normal: 1 },  // top face
                        { axis: 2, dist: pz - obs.min[2], normal: -1 }, // front face
                        { axis: 2, dist: obs.max[2] - pz, normal: 1 },  // back face
                    ];

                    // find face with smallest distance (largest penetration)
                    let closest = dists[0];
                    for (let i = 1; i < dists.length; i++) {
                        if (dists[i].dist < closest.dist) {
                            closest = dists[i];
                        }
                    }

                    // if ball is penetrating (distance < radius), resolve collision
                    if (closest.dist < b.radius) {
                        const nx = closest.axis === 0 ? closest.normal : 0;
                        const ny = closest.axis === 1 ? closest.normal : 0;
                        const nz = closest.axis === 2 ? closest.normal : 0;

                        // push ball out
                        const overlap = b.radius - closest.dist;
                        b.currentPosition[0] += nx * overlap;
                        b.currentPosition[1] += ny * overlap;
                        b.currentPosition[2] += nz * overlap;

                        // reflect velocity
                        const vDotN = b.currentVelocity[0] * nx + b.currentVelocity[1] * ny + b.currentVelocity[2] * nz;
                        if (vDotN < 0) {
                            const e = b.elasticity * obs.restitution;
                            const j = -(1 + e) * vDotN;
                            b.currentVelocity[0] += nx * j;
                            b.currentVelocity[1] += ny * j;
                            b.currentVelocity[2] += nz * j;
                        }
                    }
                }
            }
        }

        // 4. ball-ball collisions (same as basic sim)
        for (let i = 0; i < this.balls.length; i++) {
            for (let j = i + 1; j < this.balls.length; j++) {
                const b1 = this.balls[i];
                const b2 = this.balls[j];

                const dx = b2.currentPosition[0] - b1.currentPosition[0];
                const dy = b2.currentPosition[1] - b1.currentPosition[1];
                const dz = b2.currentPosition[2] - b1.currentPosition[2];
                const dist = Math.hypot(dx, dy, dz);
                const minDist = b1.radius + b2.radius;

                if (dist < minDist && dist > 0) {
                    const nx = dx / dist;
                    const ny = dy / dist;
                    const nz = dz / dist;

                    const overlap = minDist - dist;
                    const totalInvMass = b1.invMass + b2.invMass;
                    const pCorr = overlap / totalInvMass;

                    b1.currentPosition[0] -= nx * pCorr * b1.invMass;
                    b1.currentPosition[1] -= ny * pCorr * b1.invMass;
                    b1.currentPosition[2] -= nz * pCorr * b1.invMass;
                    b2.currentPosition[0] += nx * pCorr * b2.invMass;
                    b2.currentPosition[1] += ny * pCorr * b2.invMass;
                    b2.currentPosition[2] += nz * pCorr * b2.invMass;

                    const rvx = b2.currentVelocity[0] - b1.currentVelocity[0];
                    const rvy = b2.currentVelocity[1] - b1.currentVelocity[1];
                    const rvz = b2.currentVelocity[2] - b1.currentVelocity[2];
                    const vRelativeNormal = rvx * nx + rvy * ny + rvz * nz;

                    if (vRelativeNormal < 0) {
                        const e = Math.sqrt(b1.elasticity * b2.elasticity);
                        const jMag = -(1 + e) * vRelativeNormal / totalInvMass;

                        b1.currentVelocity[0] -= nx * jMag * b1.invMass;
                        b1.currentVelocity[1] -= ny * jMag * b1.invMass;
                        b1.currentVelocity[2] -= nz * jMag * b1.invMass;

                        b2.currentVelocity[0] += nx * jMag * b2.invMass;
                        b2.currentVelocity[1] += ny * jMag * b2.invMass;
                        b2.currentVelocity[2] += nz * jMag * b2.invMass;
                    }
                }
            }
        }

        // 5. ground friction and sleeping
        for (const b of this.balls) {
            const isAtRest = b.currentPosition[2] <= b.radius + 0.01;
            if (isAtRest) {
                b.currentVelocity[0] *= (1.0 - env.friction);
                b.currentVelocity[1] *= (1.0 - env.friction);
                if (Math.hypot(b.currentVelocity[0], b.currentVelocity[1]) < env.sleepThreshold) {
                    b.currentVelocity[0] = 0;
                    b.currentVelocity[1] = 0;
                }
            }
        }

        if (store) {
            const p = new Float32Array(this.balls.length * 3);
            for (let i = 0; i < this.balls.length; i++) p.set(this.balls[i].currentPosition, i * 3);
            this.positions.push(p);
        }
    }

    simulate() {
        for (const b of this.balls) b.initializeState();
        this.positions = [];
        for (let i = 0; i < this.totalTimeSteps; i++) this.step(true);
    }
}

// helper to create balls that enter from the x-min side through a tube region
export function createSideEntryBalls({
    count,
    radius,
    density,
    color = "blue",
    elasticity = 0.9,
    roomDimensions,
    entryXOffset = -0.5,
    entrySpeedX = 5.0,
    tubeYRange = null,
    tubeZRange = null,
}) {
    const balls = [];
    const [roomX, roomY, roomZ] = roomDimensions;

    const yMin = tubeYRange ? tubeYRange[0] : radius;
    const yMax = tubeYRange ? tubeYRange[1] : roomY - radius;
    const zMin = tubeZRange ? tubeZRange[0] : radius;
    const zMax = tubeZRange ? tubeZRange[1] : roomZ - radius;

    for (let i = 0; i < count; i++) {
        const y = yMin + Math.random() * Math.max(0, (yMax - yMin));
        const z = zMin + Math.random() * Math.max(0, (zMax - zMin));

        const startPosition = new Float64Array([
            entryXOffset,
            y,
            z,
        ]);

        const startVelocity = new Float64Array([
            entrySpeedX,
            0,
            0,
        ]);

        balls.push(new Ball({
            radius,
            density,
            color,
            elasticity,
            startPosition,
            startVelocity,
        }));
    }

    return balls;
}

