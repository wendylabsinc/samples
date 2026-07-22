/// The single seam between the shared flight app and the world.
/// Sim (`SimIO`) and real (`MavlinkIO`) adapters conform in later plans.
public protocol FlightIO: Sendable {
    func connect() async throws
    func arm() async throws
    func engageOffboard() async throws
    func readState() async throws -> DroneState
    func send(_ sp: AttitudeThrust) async throws
    func heartbeat() async throws
    func handback() async
    func kill() async
}
