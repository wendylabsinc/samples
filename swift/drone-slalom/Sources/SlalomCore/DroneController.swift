import Foundation
import MuJoCo

/// Geometric quadrotor controller: position error -> desired thrust axis -> reduced-attitude
/// torque -> rotor thrust mix. Ports mujoco_drone_race.py (Skydio X2).
public struct DroneController {
    // Position PD gains (per axis).
    public let kpPos = Vec3(1.1, 1.1, 10.0)
    public let kdPos = Vec3(2.2, 2.2, 6.0)
    // Attitude / yaw gains.
    public let kpAtt = 9.0, kdAtt = 1.2
    public let kpYaw = 2.0, kdYaw = 0.4
    public let g = 9.81
    // Rotor mixer geometry (from x2.xml): arm half-spans and yaw-reaction coefficient.
    public let ax = 0.14, ay = 0.18, cz = 0.0201
    public let thrustMax = 13.0

    public init() {}

    /// Mix a collective thrust T and body torques (tx,ty,tz) into 4 clipped rotor thrusts.
    public func rotorMix(_ T: Double, _ tx: Double, _ ty: Double, _ tz: Double) -> [Double] {
        let X = tx / (4 * ay), Y = ty / (4 * ax), Z = tz / (4 * cz)
        let t = [T/4 - X + Y - Z, T/4 + X + Y + Z, T/4 + X - Y - Z, T/4 - X - Y + Z]
        return t.map { Swift.min(Swift.max($0, 0.0), thrustMax) }
    }

    /// 4 rotor thrusts for the current state. `rotation` is world<-body; `angularVelocity`
    /// is body-frame; `velocity` is world-frame.
    public func control(position p: Vec3, rotation R: Mat3, velocity v: Vec3,
                        angularVelocity omega: Vec3, target tgt: Vec3, mass: Double) -> [Double] {
        let b3 = R.column(2)                       // body z-axis (thrust dir) in world
        let aDes = Vec3(kpPos.x * (tgt.x - p.x) + kdPos.x * (-v.x),
                        kpPos.y * (tgt.y - p.y) + kdPos.y * (-v.y),
                        kpPos.z * (tgt.z - p.z) + kdPos.z * (-v.z) + g)
        let T = mass * aDes.dot(b3)
        let b3des = aDes.normalized
        // Reduced-attitude control: rotate the current thrust axis toward b3des.
        let eWorld = b3.cross(b3des)
        let eBody = R.transposeTimes(eWorld)
        var tau = Vec3(kpAtt * eBody.x - kdAtt * omega.x,
                       kpAtt * eBody.y - kdAtt * omega.y,
                       kpAtt * eBody.z - kdAtt * omega.z)
        // Hold yaw ~0 (nose down +x).
        let b1 = R.column(0)
        let yaw = atan2(b1.y, b1.x)
        tau.z += -kpYaw * yaw - kdYaw * omega.z
        return rotorMix(T, tau.x, tau.y, tau.z)
    }
}
