import DroneCore

public struct ControllerGains: Sendable {
    public var kp: Double
    public var kd: Double
    public init(kp: Double, kd: Double) { self.kp = kp; self.kd = kd }
}

/// Geometric position controller. Produces an attitude+thrust setpoint —
/// the seam level. The FC (real) or SimIO's mixer model (sim) runs the
/// inner rate/rotor loop below this.
public struct GeometricController: Sendable {
    public let mass: Double
    public let gravity: Double
    public let maxThrustForce: Double
    public let gains: ControllerGains

    public init(mass: Double, gravity: Double, maxThrustForce: Double,
                gains: ControllerGains) {
        self.mass = mass; self.gravity = gravity
        self.maxThrustForce = maxThrustForce; self.gains = gains
    }

    public func compute(state: DroneState,
                        positionTarget: Vec3,
                        velocityTarget: Vec3 = .zero,
                        yawTarget: Double = 0) -> AttitudeThrust {
        // Desired acceleration: PD on position/velocity + gravity compensation.
        let ePos = positionTarget - state.position
        let eVel = velocityTarget - state.velocity
        let aDes = (ePos * gains.kp) + (eVel * gains.kd) + (Vec3.up * gravity)

        // Desired body-z aligns with the total thrust direction.
        let desiredZ = aDes.normalized()
        let attitude = Quat(desiredZ: desiredZ, yaw: yawTarget)

        // Collective thrust magnitude as a fraction of max available force.
        let force = mass * aDes.length
        let thrust = force / maxThrustForce
        return AttitudeThrust(attitude: attitude, thrust: thrust)
    }
}
