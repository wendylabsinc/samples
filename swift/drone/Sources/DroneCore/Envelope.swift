public struct Envelope: Sendable {
    public var posMin: Vec3
    public var posMax: Vec3
    public var maxTiltRadians: Double
    public var thrustMin: Double
    public var thrustMax: Double

    public init(posMin: Vec3, posMax: Vec3, maxTiltRadians: Double,
                thrustMin: Double, thrustMax: Double) {
        self.posMin = posMin; self.posMax = posMax
        self.maxTiltRadians = maxTiltRadians
        self.thrustMin = thrustMin; self.thrustMax = thrustMax
    }

    /// Soft clamp: bound thrust and limit setpoint tilt. Keeps flying.
    public func clamp(_ sp: AttitudeThrust) -> AttitudeThrust {
        let thrust = min(thrustMax, max(thrustMin, sp.thrust))
        let tilt = sp.attitude.angle(to: .up)
        guard tilt > maxTiltRadians else {
            return AttitudeThrust(attitude: sp.attitude, thrust: thrust)
        }
        // Re-derive a setpoint at the tilt limit, preserving tilt direction & yaw.
        let z = sp.attitude.bodyZ.normalized()
        // Direction of tilt in the horizontal plane.
        var horiz = Vec3(x: z.x, y: z.y, z: 0)
        if horiz.length < 1e-9 {
            return AttitudeThrust(attitude: sp.attitude, thrust: thrust)
        }
        horiz = horiz.normalized()
        let limited = (horiz * Foundation_sin(maxTiltRadians))
            + (Vec3.up * Foundation_cos(maxTiltRadians))
        // Preserve the commanded heading (yaw), not the tilt azimuth.
        let heading = sp.attitude.bodyX
        let yaw = Foundation_atan2(heading.y, heading.x)
        let q = Quat(desiredZ: limited, yaw: yaw)
        return AttitudeThrust(attitude: q, thrust: thrust)
    }

    /// Hard geofence: is the position outside the allowed box?
    public func breaches(position p: Vec3) -> Bool {
        p.x < posMin.x || p.x > posMax.x
            || p.y < posMin.y || p.y > posMax.y
            || p.z < posMin.z || p.z > posMax.z
    }
}
