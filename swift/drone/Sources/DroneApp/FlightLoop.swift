import DroneCore

/// Drives one control iteration through the seam, enforcing the loop
/// invariant: each tick either sends exactly one clamped setpoint, or hands
/// back / kills — never nothing-with-motors-live, never a stale/unclamped one.
public struct FlightLoop<IO: FlightIO>: Sendable {
    public let io: IO
    public var kernel: SafetyKernel
    public let controller: GeometricController
    public let positionTarget: Vec3
    public private(set) var lastSendTime: Double?

    public init(io: IO, kernel: SafetyKernel, controller: GeometricController,
                positionTarget: Vec3) {
        self.io = io; self.kernel = kernel
        self.controller = controller; self.positionTarget = positionTarget
    }

    /// Connect, arm, and engage offboard — advancing the adapter and the
    /// kernel in lockstep so the arm/mode gate is honored.
    public mutating func bringUp() async throws {
        try await io.connect();        kernel.didConnect()
        try await io.arm();            kernel.didArm()
        try await io.engageOffboard(); kernel.didEngageOffboard()
    }

    public mutating func tick(now: Double) async throws {
        let state = try await io.readState()
        let sendAge = lastSendTime.map { now - $0 } ?? 0
        let decision = kernel.check(position: state.position,
                                    lastUpdateAge: state.health.lastUpdateAge,
                                    sendAge: sendAge)
        switch decision {
        case .command:
            let raw = controller.compute(state: state, positionTarget: positionTarget)
            let clamped = kernel.envelope.clamp(raw)
            try await io.send(clamped)
            try await io.heartbeat()
            lastSendTime = now
        case .handback:
            await io.handback()
        case .kill:
            await io.kill()
        case .reject:
            break
        }
    }
}
