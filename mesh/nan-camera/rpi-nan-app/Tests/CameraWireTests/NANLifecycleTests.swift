import XCTest
@testable import CameraWire

final class NANLifecycleTests: XCTestCase {
    private let oldNMI = "4e:e7:7b:45:3e:6d"
    private let newNMI = "fe:68:96:ec:7b:78"
    private let pixelNDI = "ae:3e:b1:72:99:f4"
    private let otherNDI = "aa:bb:cc:dd:ee:ff"

    private func request(_ peer: String, _ id: Int, ndi: String? = nil) -> NDPRequest {
        NDPRequest(event: "<3>NAN-NDP-REQUEST peer_nmi=\(peer) ndp_id=\(id) init_ndi=\(ndi ?? pixelNDI) publish_inst_id=2", ownPublishID: "2")!
    }

    func testRequiresOurPublicationAndValidIdentifiers() {
        XCTAssertNil(NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=\(oldNMI) ndp_id=1 init_ndi=\(pixelNDI) publish_inst_id=3", ownPublishID: "2"))
        XCTAssertNil(NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=broken ndp_id=1 init_ndi=\(pixelNDI) publish_inst_id=2", ownPublishID: "2"))
        XCTAssertNil(NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=\(oldNMI) ndp_id=0 init_ndi=\(pixelNDI) publish_inst_id=2", ownPublishID: "2"))
        XCTAssertTrue(request(oldNMI, 1).response(ndi: "nan0", accept: false).hasPrefix("NAN_NDP_RESPONSE reject "))
    }

    func testHandoffWaitsForEveryOwnedDisconnectAndIgnoresDuplicates() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), second = request(oldNMI, 152), replacement = request(newNMI, 130)
        XCTAssertEqual(state.request(old, now: 0), [.accept(old)])
        XCTAssertEqual(state.event(.connected(old.ndp), now: 0), [])
        XCTAssertEqual(state.request(second, now: 0), [.accept(second)])
        XCTAssertEqual(state.request(replacement, now: 1), [.terminate(old.ndp)])
        XCTAssertEqual(state.request(replacement, now: 2), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [.terminate(second.ndp)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "152"), now: 2), [.accept(replacement)])
        XCTAssertEqual(state.request(replacement, now: 2), [])
        XCTAssertEqual(state.stop(), [.terminate(replacement.ndp)])
    }

    func testSameNMIAndDistinctNDIsDoNotTriggerHandoff() {
        var state = NDPRequestLifecycle()
        let first = request(oldNMI, 1), same = request(oldNMI, 2), other = request(newNMI, 3, ndi: otherNDI)
        XCTAssertEqual(state.request(first, now: 0), [.accept(first)])
        XCTAssertEqual(state.request(same, now: 0), [.accept(same)])
        XCTAssertEqual(state.request(other, now: 0), [.accept(other)])
        XCTAssertEqual(state.request(first, now: 0), [])
    }

    func testForeignConnectedEventsNeverGrantOwnership() {
        var state = NDPRequestLifecycle()
        let foreign = request(oldNMI, 151), replacement = request(newNMI, 130)
        XCTAssertEqual(state.event(.connected(foreign.ndp), now: 0), [])
        XCTAssertEqual(state.request(replacement, now: 1), [.accept(replacement)])
        XCTAssertEqual(state.stop(), [.terminate(replacement.ndp)])
        let raw = "NAN-NDP-CONNECTED peer=\(oldNMI) ndp_id=151 local_ndi=\(otherNDI) peer_ndi=\(pixelNDI)"
        XCTAssertNil(parseOwnedNDPEvent(raw, localNDIMAC: "00:11:22:33:44:55"))
    }

    func testDefaultDeadlineAllowsTwoHostapTerminationTimeouts() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 251), replacement = request(newNMI, 98)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.expire(now: 4), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "251"), now: 5), [.accept(replacement)])
    }

    func testTimeoutDoesNotExtendOnDuplicateOrAcceptAfterLateDisconnect() {
        var state = NDPRequestLifecycle(handoffTimeout: 3)
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        XCTAssertEqual(state.request(replacement, now: 1), [.terminate(old.ndp)])
        XCTAssertEqual(state.request(replacement, now: 3.9), [])
        XCTAssertEqual(state.request(replacement, now: 4), [.reject(replacement)])
        XCTAssertEqual(state.request(replacement, now: 4.05), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 4.1), [])
        XCTAssertEqual(state.stop(), [])
    }

    func testTimeoutWinsOverDisconnectAtDeadline() {
        var state = NDPRequestLifecycle(handoffTimeout: 3)
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 4), [.reject(replacement)])
    }

    func testOverlappingRequestRejectedWithoutReplacingFirst() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), replacement = request(newNMI, 130), overlap = request("62:19:e5:83:32:90", 243)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.request(overlap, now: 1), [.reject(overlap)])
        let oldPeerRetry = request(oldNMI, 153)
        XCTAssertEqual(state.request(oldPeerRetry, now: 1), [.reject(oldPeerRetry)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [.accept(replacement)])
    }

    func testPendingPeerCancellationDoesNotAcceptAfterOldDisconnect() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.event(.disconnected(peerNMI: newNMI, id: "130"), now: 1), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [])
    }

    func testTerminationFailureRejectsReplacementAndKeepsOldOwnedForCleanup() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.commandFailed(.terminate(old.ndp)), [.reject(replacement)])
        XCTAssertEqual(state.stop(), [.terminate(old.ndp)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [])
    }

    func testAmbiguousAcceptFailureTerminatesOwnedAttempt() {
        var state = NDPRequestLifecycle()
        let attempt = request(oldNMI, 151)
        _ = state.request(attempt, now: 0)
        XCTAssertEqual(state.commandFailed(.accept(attempt)), [.terminate(attempt.ndp)])
        XCTAssertEqual(state.commandFailed(.terminate(attempt.ndp)), [])
        XCTAssertEqual(state.stop(), [.terminate(attempt.ndp)])
    }

    func testIndependentPendingPeersAndCapacityAreBounded() {
        var state = NDPRequestLifecycle(maximumOwned: 3, maximumPending: 1)
        let a = request(oldNMI, 1), b = request(oldNMI, 2, ndi: otherNDI)
        let newA = request(newNMI, 3), newB = request(newNMI, 4, ndi: otherNDI)
        _ = state.request(a, now: 0)
        _ = state.request(b, now: 0)
        XCTAssertEqual(state.request(newA, now: 1), [.terminate(a.ndp)])
        XCTAssertEqual(state.request(newB, now: 1), [.reject(newB)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "2"), now: 1), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 2), [.accept(newA)])

        var full = NDPRequestLifecycle(maximumOwned: 1)
        _ = full.request(a, now: 0)
        XCTAssertEqual(full.request(b, now: 0), [.reject(b)])
    }

    func testTwoPendingNDIsSerializeTerminationGloballyPerOldNMI() {
        var state = NDPRequestLifecycle()
        let a = request(oldNMI, 1), b = request(oldNMI, 2, ndi: otherNDI)
        let newA = request(newNMI, 3), newB = request("62:19:e5:83:32:90", 4, ndi: otherNDI)
        _ = state.request(a, now: 0); _ = state.request(b, now: 0)
        XCTAssertEqual(state.request(newA, now: 1), [.terminate(a.ndp)])
        XCTAssertEqual(state.request(newB, now: 1), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 5), [.accept(newA), .terminate(b.ndp)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "2"), now: 9), [.accept(newB)])
    }

    func testDuplicateKeyCannotMoveBetweenPendingNDIs() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        let changed = request(newNMI, 130, ndi: otherNDI)
        XCTAssertEqual(state.request(changed, now: 1), [.reject(changed)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [.accept(replacement)])
    }

    func testStopRejectsPendingAndNeverAcceptsLateEvents() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 151), replacement = request(newNMI, 130)
        _ = state.request(old, now: 0)
        _ = state.request(replacement, now: 1)
        XCTAssertEqual(state.stop(), [.reject(replacement), .terminate(old.ndp)])
        XCTAssertEqual(state.stop(), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "151"), now: 2), [])
        XCTAssertEqual(state.request(replacement, now: 2), [.reject(replacement)])
    }

    func testMultipleHandlesAllowSequentialFourSecondHostapCleanup() {
        var state = NDPRequestLifecycle()
        let a = request(oldNMI, 1), b = request(oldNMI, 2), next = request(newNMI, 3)
        _ = state.request(a, now: 0); _ = state.request(b, now: 0)
        XCTAssertEqual(state.request(next, now: 1), [.terminate(a.ndp)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 5), [.terminate(b.ndp)])
        XCTAssertEqual(state.expire(now: 8), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "2"), now: 9), [.accept(next)])
    }

    func testLongChainIsBoundedBelowHostapSetupDeadline() {
        var state = NDPRequestLifecycle()
        for id in 1...5 { _ = state.request(request(oldNMI, id), now: 0) }
        let next = request(newNMI, 6)
        XCTAssertEqual(state.request(next, now: 1), [.terminate(request(oldNMI, 1).ndp)])
        for id in 1...4 {
            XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: String(id)), now: 1 + Double(id) * 4), [.terminate(request(oldNMI, id + 1).ndp)])
        }
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "5"), now: 21), [.reject(next)])
    }

    func testDifferentOldNMIsCanTerminateInParallel() {
        var state = NDPRequestLifecycle()
        let a = request(oldNMI, 1), b = request(newNMI, 2, ndi: otherNDI)
        let nextA = request("62:19:e5:83:32:90", 3), nextB = request("86:1b:f0:32:fd:b4", 4, ndi: otherNDI)
        _ = state.request(a, now: 0); _ = state.request(b, now: 0)
        XCTAssertEqual(state.request(nextA, now: 1), [.terminate(a.ndp)])
        XCTAssertEqual(state.request(nextB, now: 1), [.terminate(b.ndp)])
    }

    private func failedDisconnect(_ peer: String, _ id: Int, local: String = "00:00:00:00:00:00") -> String {
        "NAN-NDP-DISCONNECTED peer=\(peer) ndp_id=\(id) local_ndi=\(local) peer_ndi=\(pixelNDI) failure=1"
    }

    func testZeroLocalDisconnectReleasesOnlyAnAlreadyOwnedRequest() {
        var state = NDPRequestLifecycle(maximumOwned: 1)
        let owned = request(oldNMI, 1), next = request(newNMI, 2)
        _ = state.request(owned, now: 0)
        let unrelated = failedDisconnect(newNMI, 2)
        XCTAssertNil(parseOwnedNDPEvent(unrelated, localNDIMAC: otherNDI))
        XCTAssertEqual(state.event(raw: unrelated, localNDIMAC: otherNDI, now: 1), [])
        XCTAssertEqual(state.request(next, now: 1), [.reject(next)])
        XCTAssertEqual(state.event(raw: failedDisconnect(oldNMI, 1, local: pixelNDI), localNDIMAC: otherNDI, now: 1), [])
        XCTAssertEqual(state.request(next, now: 1), [.reject(next)])
        XCTAssertEqual(state.event(raw: failedDisconnect(oldNMI, 1), localNDIMAC: otherNDI, now: 2), [])
        XCTAssertEqual(state.request(next, now: 2), [.accept(next)])
    }

    func testZeroLocalDisconnectCancelsOnlyTheMatchingPendingRequest() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 1), next = request(newNMI, 2)
        _ = state.request(old, now: 0); _ = state.request(next, now: 1)
        XCTAssertEqual(state.event(raw: failedDisconnect(newNMI, 999), localNDIMAC: otherNDI, now: 2), [])
        XCTAssertEqual(state.event(raw: failedDisconnect(newNMI, 2), localNDIMAC: otherNDI, now: 2), [])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 3), [])
        XCTAssertEqual(state.stop(), [])
    }

    func testZeroLocalOwnedDisconnectAdvancesSerializedHandoff() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 1), next = request(newNMI, 2)
        _ = state.request(old, now: 0); _ = state.request(next, now: 1)
        XCTAssertEqual(state.event(raw: failedDisconnect(oldNMI, 1), localNDIMAC: otherNDI, now: 2), [.accept(next)])
    }

    func testNewRequestCannotStartOnTerminatingNMI() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 1), next = request(newNMI, 2)
        _ = state.request(old, now: 0)
        XCTAssertEqual(state.request(next, now: 1), [.terminate(old.ndp)])
        let busy = request(oldNMI, 3, ndi: otherNDI)
        XCTAssertEqual(state.request(busy, now: 2), [.reject(busy)])
        let independent = request("62:19:e5:83:32:90", 4, ndi: otherNDI)
        XCTAssertEqual(state.request(independent, now: 2), [.accept(independent)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 3), [.accept(next)])
    }

    func testAmbiguousAcceptCleanupKeepsNMITransactionBusy() {
        var state = NDPRequestLifecycle()
        let old = request(oldNMI, 1), next = request(oldNMI, 2, ndi: otherNDI)
        XCTAssertEqual(state.request(old, now: 0), [.accept(old)])
        XCTAssertEqual(state.commandFailed(.accept(old)), [.terminate(old.ndp)])
        XCTAssertEqual(state.request(next, now: 1), [.reject(next)])
        XCTAssertEqual(state.event(.disconnected(peerNMI: oldNMI, id: "1"), now: 2), [])
        XCTAssertEqual(state.request(next, now: 2), [.accept(next)])
    }

}
