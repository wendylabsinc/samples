import Testing
@testable import DroneCore

@Test func vec3CrossAndDot() {
    let x = Vec3(x: 1, y: 0, z: 0)
    let y = Vec3(x: 0, y: 1, z: 0)
    #expect(x.cross(y) == Vec3(x: 0, y: 0, z: 1))
    #expect(x.dot(y) == 0)
    #expect(x.dot(x) == 1)
}

@Test func vec3Normalize() {
    let v = Vec3(x: 0, y: 0, z: 5).normalized()
    #expect(abs(v.length - 1) < 1e-12)
    #expect(abs(v.z - 1) < 1e-12)
}

@Test func quatFromDesiredZLevelIsIdentityUp() {
    // Desired body-z pointing straight up, yaw 0 → level attitude.
    let q = Quat(desiredZ: .up, yaw: 0).normalized()
    #expect(abs(q.length - 1) < 1e-9)
    #expect(q.angle(to: .up) < 1e-6)   // body-z coincides with world up
}

@Test func quatFromDesiredZTilted() {
    // Desired body-z tilted toward +x → body-z axis has positive x.
    let desired = Vec3(x: 0.3, y: 0, z: 1).normalized()
    let q = Quat(desiredZ: desired, yaw: 0)
    #expect(q.angle(to: .up) > 0.1)    // tilted away from vertical
}
