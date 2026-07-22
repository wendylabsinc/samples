import Testing
@testable import DroneCore

private let env = Envelope(
    posMin: Vec3(x: -10, y: -10, z: 0),
    posMax: Vec3(x: 10, y: 10, z: 20),
    maxTiltRadians: 0.5,          // ~28.6°
    thrustMin: 0.05,
    thrustMax: 0.9
)

@Test func clampLimitsThrust() {
    let hot = AttitudeThrust(attitude: .identity, thrust: 2.0)
    #expect(env.clamp(hot).thrust == 0.9)
    let cold = AttitudeThrust(attitude: .identity, thrust: -1.0)
    #expect(env.clamp(cold).thrust == 0.05)
}

@Test func clampLimitsTilt() {
    // Desired body-z tilted 60° from vertical — must be clamped to ~28.6°.
    let steep = Quat(desiredZ: Vec3(x: 1, y: 0, z: 0.577), yaw: 0) // ~60°
    let sp = AttitudeThrust(attitude: steep, thrust: 0.5)
    let clamped = env.clamp(sp)
    #expect(clamped.attitude.angle(to: .up) <= 0.5 + 1e-6)
    #expect(clamped.attitude.angle(to: .up) > 0.4)  // clamped to the limit, not zeroed
}

@Test func clampLeavesGentleTiltUntouched() {
    let gentle = Quat(desiredZ: Vec3(x: 0.1, y: 0, z: 1), yaw: 0) // ~5.7°
    let sp = AttitudeThrust(attitude: gentle, thrust: 0.5)
    let clamped = env.clamp(sp)
    #expect(abs(clamped.attitude.angle(to: .up) - gentle.angle(to: .up)) < 1e-6)
}

@Test func breachesDetectsGeofence() {
    #expect(env.breaches(position: Vec3(x: 0, y: 0, z: 5)) == false)
    #expect(env.breaches(position: Vec3(x: 11, y: 0, z: 5)) == true)   // x past max
    #expect(env.breaches(position: Vec3(x: 0, y: 0, z: -1)) == true)   // below floor
}
