import Foundation

/// Process-owned discovery leases. All times use system uptime, matching
/// hostap's monotonic TTL. Commands and this state are serialized by the caller.
/// Cancel ownership is consumed before sending: a lost reply must not cause a
/// later retry to cancel a handle already reused by another application.
public struct NANPublicationLease {
    public static let ttl: TimeInterval = 180
    public static let refresh: TimeInterval = 120
    /// Unknown publish IDs cannot safely be canceled; do not immediately create
    /// another lease after a lost response. Keep the process/event loop alive.
    public static let uncertainRetryDelay: TimeInterval = ttl + 3
    // Both handles accept while advertised. Admitted open requests retain
    // their own bounded authorization across retirement of the original ID.
    private static let retireGrace: TimeInterval = 35
    private static let commandMargin: TimeInterval = 3
    private struct Owned {
        let expires: TimeInterval
        let generation: UInt64
        var cancelAt: TimeInterval
    }
    private struct Admission {
        let generation: UInt64
        let deadline: TimeInterval
        var consumed = false
    }
    private var admissions: [NDPRequest: Admission] = [:]
    private var owned: [String: Owned] = [:]
    private var current: String?
    private var refreshAt: TimeInterval = 0
    private var nextAttempt: TimeInterval = 0
    private var attempt: (token: UInt64, started: TimeInterval)?
    private var serial: UInt64 = 0
    private var stopped = false

    public init() {}

    public init(initialID: String, started: TimeInterval) {
        current = initialID
        owned[initialID] = Owned(expires: started + Self.ttl,
                                 generation: 0, cancelAt: .infinity)
        refreshAt = started + Self.refresh
    }

    private mutating func discardExpired(now: TimeInterval) {
        owned = owned.filter { $0.value.expires > now }
        if let current, owned[current] == nil { self.current = nil }
    }

    public mutating func acceptingIDs(now: TimeInterval) -> [String] {
        discardExpired(now: now)
        return stopped ? [] : owned.keys.sorted()
    }

    /// Only the fixed, open camera service may use this authorization. Each
    /// request must first name an owned advertised publication incarnation;
    /// numeric ID reuse or a later retransmission cannot extend its deadline.
    public mutating func admitOpenRequest(_ request: NDPRequest, now: TimeInterval) -> Bool {
        discardExpired(now: now)
        admissions = admissions.filter { $0.value.deadline + 30 > now }
        guard !stopped, request.cipherSuite == 0,
              let original = owned[request.publishID], now + Self.commandMargin < original.expires else { return false }
        if let admission = admissions[request] {
            return now < admission.deadline && admission.generation == original.generation
        }
        guard admissions.count < 64 else { return false }
        admissions[request] = Admission(generation: original.generation, deadline: now + 30)
        return true
    }

    /// The control API handle selects local service/security context; it does
    /// not replace the pending NDP tuple or its original on-air publication ID.
    /// All publications in this instance are the same fixed open camera service.
    /// Consume authorization before I/O: an ambiguous response must not replay.
    public mutating func openResponse(_ request: NDPRequest, ndi: String, now: TimeInterval) -> String? {
        discardExpired(now: now)
        guard !stopped, var admission = admissions[request], !admission.consumed else { return nil }
        admission.consumed = true
        admissions[request] = admission
        guard now + Self.commandMargin < admission.deadline,
              let current, let active = owned[current],
              now + Self.commandMargin < active.expires else { return nil }
        return request.response(ndi: ndi, accept: true, serviceHandle: current)
    }

    public mutating func finishRequest(_ request: NDPRequest) {
        if var admission = admissions[request] {
            admission.consumed = true
            admissions[request] = admission
        }
    }

    public mutating func beginRefresh(now: TimeInterval) -> UInt64? {
        discardExpired(now: now)
        guard !stopped, attempt == nil, now >= nextAttempt,
              current == nil || now >= refreshAt else { return nil }
        serial &+= 1
        attempt = (serial, now)
        return serial
    }

    /// Returns a late successful publish to cancel if Stop won the race.
    /// A nil ID keeps the current lease. Definitive failures retry after five
    /// seconds; unknown outcomes defer by uncertainRetryDelay without blocking.
    public mutating func completeRefresh(token: UInt64, id: String?, now: TimeInterval, outcomeUnknown: Bool = false) -> [String] {
        guard let attempt, attempt.token == token else { return [] }
        self.attempt = nil
        nextAttempt = now + (outcomeUnknown ? Self.uncertainRetryDelay : 5)
        discardExpired(now: now)
        guard let id else { return [] }
        let expires = attempt.started + Self.ttl
        guard now + Self.commandMargin < expires else { return [] }
        if stopped { return [id] }
        if let current, current != id, var old = owned[current] {
            old.cancelAt = min(old.expires, now + Self.retireGrace)
            owned[current] = old
        }
        owned[id] = Owned(expires: expires, generation: token, cancelAt: .infinity)
        current = id
        refreshAt = attempt.started + Self.refresh
        return []
    }

    public mutating func cancellations(now: TimeInterval) -> [String] {
        discardExpired(now: now)
        var ids: [String] = []
        for (id, lease) in owned where now >= lease.cancelAt {
            // At or near expiry the handle may be freed/reused before a
            // control command is processed. Let hostap expire it instead.
            if now + Self.commandMargin < lease.expires { ids.append(id) }
            owned.removeValue(forKey: id)
        }
        return ids.sorted()
    }

    public mutating func terminated(id: String) {
        owned.removeValue(forKey: id)
        if current == id { current = nil }
    }

    public mutating func stop(now: TimeInterval) -> [String] {
        guard !stopped else { return [] }
        stopped = true
        let ids = owned.filter { now + Self.commandMargin < $0.value.expires }.keys.sorted()
        owned.removeAll()
        admissions.removeAll()
        current = nil
        return ids
    }
}
