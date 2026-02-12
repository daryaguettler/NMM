export class Ball {
    constructor({ radius, density, color = "blue", elasticity, startPosition, startVelocity }) {
        this.radius = radius;
        this.density = density;
        this.color = color;
        this.elasticity = elasticity; // This is the 'restitution' of the ball's material
        
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

export class BounceEnvironment {
    constructor({
        roomDimensions,
        gravity,
        gravityVector = [0, 0, -1],
        wallRestitution = 0.9,
        friction = 0.1,
        sleepThreshold = 0.05,
        fluidDensity = 1.225,
        dragCoefficient = 0.47,
        magnusCoefficient = 0.5,
        angularDragCoefficient = 0.1,
        wallFriction = 0.2,
        ballFriction = 0.3,
        rollingFriction = 0.3,
        contactTolerance = 0.05,
    }) {
        this.roomDimensions = roomDimensions.slice();
        this.gravity = gravity;
        this.gravityVector = gravityVector.slice();
        this.wallRestitution = wallRestitution;

        // scalar friction used in simple ground slow-down
        this.friction = friction ?? rollingFriction ?? 0.1;
        this.sleepThreshold = sleepThreshold;

        // store extra parameters for possible future use
        this.fluidDensity = fluidDensity;
        this.dragCoefficient = dragCoefficient;
        this.magnusCoefficient = magnusCoefficient;
        this.angularDragCoefficient = angularDragCoefficient;
        this.wallFriction = wallFriction;
        this.ballFriction = ballFriction;
        this.rollingFriction = rollingFriction;
        this.contactTolerance = contactTolerance;

        // precompute gravity vector used for integration
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

export class Simulation {
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

        // 1. INTEGRATION (Move balls)
        for (const b of this.balls) {
            b.currentVelocity[0] += env.gNormalized[0] * dt;
            b.currentVelocity[1] += env.gNormalized[1] * dt;
            b.currentVelocity[2] += env.gNormalized[2] * dt;

            b.currentPosition[0] += b.currentVelocity[0] * dt;
            b.currentPosition[1] += b.currentVelocity[1] * dt;
            b.currentPosition[2] += b.currentVelocity[2] * dt;
        }

        // 2. WALL COLLISIONS (With Restitution)
        for (const b of this.balls) {
            for (let i = 0; i < 3; i++) {
                const minB = b.radius;
                const maxB = env.roomDimensions[i] - b.radius;
                
                // Effective restitution for the wall
                const wall_e = b.elasticity * env.wallRestitution;

                if (b.currentPosition[i] < minB) {
                    b.currentPosition[i] = minB;
                    if (b.currentVelocity[i] < 0) b.currentVelocity[i] *= -wall_e;
                } else if (b.currentPosition[i] > maxB) {
                    b.currentPosition[i] = maxB;
                    if (b.currentVelocity[i] > 0) b.currentVelocity[i] *= -wall_e;
                }
            }
        }

        // 3. BALL-BALL COLLISIONS (With Combined Restitution)
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
                    // Normal vector of collision
                    const nx = dx / dist, ny = dy / dist, nz = dz / dist;
                    
                    // a. Positional Correction (Prevents "sticking" or merging)
                    const overlap = minDist - dist;
                    const totalInvMass = b1.invMass + b2.invMass;
                    const pCorr = overlap / totalInvMass;
                    
                    b1.currentPosition[0] -= nx * pCorr * b1.invMass;
                    b1.currentPosition[1] -= ny * pCorr * b1.invMass;
                    b1.currentPosition[2] -= nz * pCorr * b1.invMass;
                    b2.currentPosition[0] += nx * pCorr * b2.invMass;
                    b2.currentPosition[1] += ny * pCorr * b2.invMass;
                    b2.currentPosition[2] += nz * pCorr * b2.invMass;

                    // b. Velocity Reflection (Impulse)
                    const rvx = b2.currentVelocity[0] - b1.currentVelocity[0];
                    const rvy = b2.currentVelocity[1] - b1.currentVelocity[1];
                    const rvz = b2.currentVelocity[2] - b1.currentVelocity[2];
                    const vRelativeNormal = rvx * nx + rvy * ny + rvz * nz;

                    // Only resolve if they are moving TOWARD each other
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
        
        // Final "Anti-Slide" Friction & Sleeping
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