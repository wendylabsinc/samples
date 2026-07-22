import DroneCore

/// Deterministic, programmable FlightIO for tests. Records calls and returns
/// a caller-supplied DroneState. Test-only; single-threaded use, hence
/// @unchecked Sendable.
final class FakeFlightIO: FlightIO, @unchecked Sendable {
    var nextState: DroneState
    var connectCalls = 0
    var armCalls = 0
    var offboardCalls = 0
    var sent: [AttitudeThrust] = []
    var heartbeats = 0
    var handbackCalled = false
    var killCalled = false

    init(nextState: DroneState) { self.nextState = nextState }

    func connect() async throws { connectCalls += 1 }
    func arm() async throws { armCalls += 1 }
    func engageOffboard() async throws { offboardCalls += 1 }
    func readState() async throws -> DroneState { nextState }
    func send(_ sp: AttitudeThrust) async throws { sent.append(sp) }
    func heartbeat() async throws { heartbeats += 1 }
    func handback() async { handbackCalled = true }
    func kill() async { killCalled = true }
}

extension DroneState {
    /// Convenience builder for tests.
    static func at(position: Vec3, lastUpdateAge: Double) -> DroneState {
        DroneState(
            t: 0, position: position, velocity: .zero,
            attitude: .identity, bodyRates: .zero,
            health: LinkHealth(armed: true, mode: .offboard,
                               lastUpdateAge: lastUpdateAge)
        )
    }
}
