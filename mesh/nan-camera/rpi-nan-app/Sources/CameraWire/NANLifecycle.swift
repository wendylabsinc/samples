import Foundation

public struct OwnedNDP: Equatable, Hashable {
    public let peerNMI: String
    public let initiatorNDI: String
    public let id: String
    public var key: String { peerNMI.lowercased() + "#" + id }

    public init(peerNMI: String, initiatorNDI: String, id: String) {
        self.peerNMI = peerNMI.lowercased()
        self.initiatorNDI = initiatorNDI.lowercased()
        self.id = id
    }

    public var terminateCommand: String {
        "NAN_NDP_TERMINATE peer_nmi=\(peerNMI) init_ndi=\(initiatorNDI) ndp_id=\(id)"
    }
}

public enum OwnedNDPEvent: Equatable {
    case connected(OwnedNDP)
    case disconnected(peerNMI: String, id: String)
}

/// wpa_supplicant emits MAC addresses in local_ndi/peer_ndi, not Linux device
/// names. A responder's peer_ndi is the initiator NDI needed for TERMINATE.
public func parseOwnedNDPEvent(_ raw: String, localNDIMAC: String,
                               unassignedDisconnectKeys: Set<String> = []) -> OwnedNDPEvent? {
    let fields = Dictionary(raw.split(separator: " ").compactMap { field -> (String, String)? in
        guard let equals = field.firstIndex(of: "=") else { return nil }
        return (String(field[..<equals]), String(field[field.index(after: equals)...]))
    }, uniquingKeysWith: { first, _ in first })
    guard let peer = fields["peer"], isMAC(peer),
          let id = fields["ndp_id"], let numericID = UInt32(id), numericID > 0 else { return nil }
    if raw.contains("NAN-NDP-CONNECTED") {
        guard fields["local_ndi"]?.lowercased() == localNDIMAC.lowercased(),
              let initiator = fields["peer_ndi"], isMAC(initiator) else { return nil }
        return .connected(OwnedNDP(peerNMI: peer, initiatorNDI: initiator, id: id))
    }
    if raw.contains("NAN-NDP-DISCONNECTED") {
        if let local = fields["local_ndi"], !local.isEmpty,
           local.lowercased() != localNDIMAC.lowercased() {
            // Failed setup can finish before hostap assigns a local NDI. Only
            // an exact request we already own or await can release that state.
            guard local == "00:00:00:00:00:00",
                  unassignedDisconnectKeys.contains(peer.lowercased() + "#" + String(numericID)) else { return nil }
        }
        return .disconnected(peerNMI: peer, id: String(numericID))
    }
    return nil
}

private func isMAC(_ value: String) -> Bool {
    let parts = value.split(separator: ":")
    return parts.count == 6 && parts.allSatisfy { $0.count == 2 && UInt8($0, radix: 16) != nil }
}


public struct NDPRequest: Equatable, Hashable {
    public let peerNMI: String
    public let ndpID: String
    public let initiatorNDI: String
    public let publishID: String
    public let cipherSuite: UInt32?
    public var ndp: OwnedNDP { OwnedNDP(peerNMI: peerNMI, initiatorNDI: initiatorNDI, id: ndpID) }

    public init?(event: String, ownPublishID: String) {
        guard event.contains("NAN-NDP-REQUEST") else { return nil }
        let values = Dictionary(event.split(separator: " ").compactMap { field -> (String, String)? in
            guard let equals = field.firstIndex(of: "=") else { return nil }
            return (String(field[..<equals]), String(field[field.index(after: equals)...]))
        }, uniquingKeysWith: { first, _ in first })
        guard values["publish_inst_id"] == ownPublishID,
              let publish = UInt32(ownPublishID), publish > 0,
              let nmi = values["peer_nmi"], isMAC(nmi),
              let ndi = values["init_ndi"], isMAC(ndi),
              let id = values["ndp_id"], let numericID = UInt32(id), numericID > 0 else { return nil }
        peerNMI = nmi.lowercased(); initiatorNDI = ndi.lowercased()
        ndpID = String(numericID); publishID = ownPublishID
        cipherSuite = values["csid"].flatMap(UInt32.init)
    }

    public func response(ndi: String, accept: Bool, serviceHandle: String? = nil) -> String {
        "NAN_NDP_RESPONSE \(accept ? "accept" : "reject") handle=\(serviceHandle ?? publishID) ndi=\(ndi) peer_nmi=\(peerNMI) ndp_id=\(ndpID) init_ndi=\(initiatorNDI)"
    }
}

public enum NDPAction: Equatable {
    case accept(NDPRequest)
    case reject(NDPRequest)
    case terminate(OwnedNDP)
}

/// The caller serializes this state and executes its command actions. A Pixel
/// can keep its NDI MAC across a randomized NMI. The kernel station is bound to
/// the old NMI, so a replacement must wait for all our old NDPs to disconnect.
/// We never infer ownership from another application's CONNECTED event.
public struct NDPRequestLifecycle {
    private struct Pending {
        let request: NDPRequest
        var waiting: Set<String>
        let deadline: TimeInterval
    }
    private var owned: [String: OwnedNDP] = [:]
    private var pending: [String: Pending] = [:] // One replacement per peer NDI.
    // hostap has one termination transaction per NMI, even across peer NDIs.
    private var terminating: [String: String] = [:] // NMI -> owned NDP key.
    // Supplicant's request setup lifetime is 30 seconds. Remember timed-out
    // requests for that window so a retransmission cannot restart the wait.
    private var expiredRequests: [String: TimeInterval] = [:]
    private var stopped = false
    private let handoffTimeout: TimeInterval
    private let maximumOwned: Int
    private let maximumPending: Int

    // A vanished old NMI takes two 2s hostap termination timeouts on the
    // tested Pi/Pixel path. Six seconds permits that cleanup plus event slack.
    public init(handoffTimeout: TimeInterval = 6, maximumOwned: Int = 32, maximumPending: Int = 8) {
        precondition(handoffTimeout > 0 && maximumOwned > 0 && maximumPending > 0)
        self.handoffTimeout = handoffTimeout
        self.maximumOwned = maximumOwned
        self.maximumPending = maximumPending
    }

    /// Retransmissions of our existing tuple must not be rejected merely
    /// because its original discovery handle has retired or permission was used.
    public func knowsRequest(_ request: NDPRequest) -> Bool {
        owned[request.ndp.key] == request.ndp || pending.values.contains { $0.request.ndp == request.ndp }
    }

    public mutating func request(_ request: NDPRequest, now: TimeInterval) -> [NDPAction] {
        var actions = expire(now: now)
        guard !stopped else { return actions + [.reject(request)] }
        guard expiredRequests[request.ndp.key] == nil else { return actions }
        if let existing = owned[request.ndp.key] {
            return existing == request.ndp ? actions : actions + [.reject(request)]
        }
        if let existing = pending.values.first(where: { $0.request.ndp.key == request.ndp.key }) {
            return existing.request == request ? actions : actions + [.reject(request)]
        }
        if let existing = pending[request.initiatorNDI] {
            // Retransmissions must not extend the deadline or terminate twice.
            return existing.request == request ? actions : actions + [.reject(request)]
        }
        // hostap cannot start another NDP transaction on a terminating NMI.
        guard terminating[request.peerNMI] == nil else { return actions + [.reject(request)] }
        guard owned.count < maximumOwned else { return actions + [.reject(request)] }
        let conflicts = owned.values.filter {
            $0.initiatorNDI == request.initiatorNDI && $0.peerNMI != request.peerNMI
        }.sorted { $0.key < $1.key }
        if conflicts.isEmpty {
            owned[request.ndp.key] = request.ndp
            actions.append(.accept(request))
        } else if pending.count < maximumPending {
            let waiting = Set(conflicts.map(\.key))
            let queued = Set(pending.values.flatMap { $0.waiting }).union(waiting)
            let longestChain = Dictionary(grouping: queued.compactMap { owned[$0] }, by: \.peerNMI)
                .values.map(\.count).max() ?? 1
            // A stale peer needs two 2-second hostap termination timeouts per
            // handle. Bound the whole handoff below its 30-second setup timer.
            let budget = min(20, handoffTimeout * Double(longestChain))
            pending[request.initiatorNDI] = Pending(request: request,
                waiting: waiting, deadline: now + budget)
            actions += scheduleTerminations()
        } else {
            actions.append(.reject(request))
        }
        return actions
    }

    private mutating func scheduleTerminations() -> [NDPAction] {
        let waiting = Set(pending.values.flatMap { $0.waiting })
        var actions: [NDPAction] = []
        for key in waiting.sorted() {
            guard let ndp = owned[key], terminating[ndp.peerNMI] == nil else { continue }
            terminating[ndp.peerNMI] = key
            actions.append(.terminate(ndp))
        }
        return actions
    }

    public mutating func event(raw: String, localNDIMAC: String, now: TimeInterval) -> [NDPAction] {
        let known = Set(owned.keys).union(pending.values.map { $0.request.ndp.key })
        guard let parsed = parseOwnedNDPEvent(raw, localNDIMAC: localNDIMAC,
                                             unassignedDisconnectKeys: known) else { return expire(now: now) }
        return event(parsed, now: now)
    }

    public mutating func event(_ event: OwnedNDPEvent, now: TimeInterval) -> [NDPAction] {
        var actions = expire(now: now)
        guard !stopped else { return actions }
        switch event {
        case .connected:
            // Ownership was recorded before sending ACCEPT; unrelated events
            // on this control socket never grant us permission to terminate.
            break
        case .disconnected(let peer, let id):
            let key = peer.lowercased() + "#" + id
            owned.removeValue(forKey: key)
            if terminating[peer.lowercased()] == key { terminating.removeValue(forKey: peer.lowercased()) }
            for ndi in pending.keys.sorted() {
                guard var replacement = pending[ndi] else { continue }
                if replacement.request.ndp.key == key {
                    // The new peer already abandoned its request.
                    pending.removeValue(forKey: ndi)
                    continue
                }
                replacement.waiting.remove(key)
                if replacement.waiting.isEmpty {
                    pending.removeValue(forKey: ndi)
                    owned[replacement.request.ndp.key] = replacement.request.ndp
                    actions.append(.accept(replacement.request))
                } else {
                    pending[ndi] = replacement
                }
            }
        }
        actions += scheduleTerminations()
        return actions
    }

    public mutating func expire(now: TimeInterval) -> [NDPAction] {
        expiredRequests = expiredRequests.filter { $0.value > now }
        let expired = pending.keys.sorted().filter { pending[$0]!.deadline <= now }
        return expired.compactMap { ndi in
            guard let request = pending.removeValue(forKey: ndi)?.request else { return nil }
            if expiredRequests.count >= 64, let oldest = expiredRequests.min(by: { $0.value < $1.value })?.key {
                expiredRequests.removeValue(forKey: oldest)
            }
            expiredRequests[request.ndp.key] = now + 30
            return .reject(request)
        }
    }

    /// The caller rejected an ACCEPT action before sending any acceptance.
    /// Release only the exact owned tuple; no radio termination is required.
    public mutating func responseNotAccepted(_ request: NDPRequest) {
        if owned[request.ndp.key] == request.ndp {
            owned.removeValue(forKey: request.ndp.key)
        }
    }

    public mutating func commandFailed(_ action: NDPAction) -> [NDPAction] {
        switch action {
        case .accept(let request):
            // A command timeout is ambiguous: ACCEPT might have succeeded.
            // Retain ownership until DISCONNECTED and attempt safe cleanup.
            terminating[request.peerNMI] = request.ndp.key
            return [.terminate(request.ndp)]
        case .terminate(let ndp):
            if terminating[ndp.peerNMI] == ndp.key { terminating.removeValue(forKey: ndp.peerNMI) }
            let affected = pending.keys.sorted().filter { pending[$0]!.waiting.contains(ndp.key) }
            return affected.compactMap { ndi in
                pending.removeValue(forKey: ndi).map { .reject($0.request) }
            }
        case .reject:
            return []
        }
    }

    public mutating func stop() -> [NDPAction] {
        guard !stopped else { return [] }
        stopped = true
        let actions = pending.keys.sorted().map { NDPAction.reject(pending[$0]!.request) }
            + owned.keys.sorted().map { NDPAction.terminate(owned[$0]!) }
        pending.removeAll()
        owned.removeAll()
        terminating.removeAll()
        expiredRequests.removeAll()
        return actions
    }
}
