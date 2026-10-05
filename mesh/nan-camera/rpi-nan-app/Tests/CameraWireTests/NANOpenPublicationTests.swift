import XCTest
@testable import CameraWire

final class NANOpenPublicationTests: XCTestCase {
    private func request(_ id: String = "1", peer: String = "01:02:03:04:05:06", ndp: String = "7", csid: String = "0") -> NDPRequest {
        NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=\(peer) ndp_id=\(ndp) init_ndi=aa:bb:cc:dd:ee:ff publish_inst_id=\(id) csid=\(csid)", ownPublishID: id)!
    }
    private func refresh(_ lease: inout NANPublicationLease, id: String, at time: Double) {
        let token = lease.beginRefresh(now: time)!
        XCTAssertEqual(lease.completeRefresh(token: token, id: id, now: time), [])
    }

    func testLateOverlapRequestAcceptedAndUsesLiveSameServiceContext() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        refresh(&lease, id: "2", at: 120)
        let old = request()
        XCTAssertTrue(lease.acceptingIDs(now: 140).contains("1"))
        XCTAssertTrue(lease.admitOpenRequest(old, now: 140))
        XCTAssertEqual(lease.openResponse(old, ndi: "camera", now: 141), old.response(ndi: "camera", accept: true, serviceHandle: "2"))
        XCTAssertEqual(old.publishID, "1") // The admitted request is not rewritten.
        XCTAssertNil(lease.openResponse(old, ndi: "camera", now: 142))
    }

    func testDeferredHandoffSurvivesOriginalPublicationRetirement() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        var lifecycle = NDPRequestLifecycle()
        let oldPeer = request(peer: "11:12:13:14:15:16", ndp: "4")
        XCTAssertEqual(lifecycle.request(oldPeer, now: 1), [.accept(oldPeer)])
        refresh(&lease, id: "2", at: 120)
        let replacement = request()
        XCTAssertTrue(lease.admitOpenRequest(replacement, now: 154))
        XCTAssertEqual(lifecycle.request(replacement, now: 154), [.terminate(oldPeer.ndp)])
        XCTAssertEqual(lease.cancellations(now: 155), ["1"])
        XCTAssertEqual(lifecycle.event(.disconnected(peerNMI: oldPeer.peerNMI, id: oldPeer.ndpID), now: 158), [.accept(replacement)])
        XCTAssertEqual(lease.openResponse(replacement, ndi: "camera", now: 158), replacement.response(ndi: "camera", accept: true, serviceHandle: "2"))
        XCTAssertFalse(lease.admitOpenRequest(request(ndp: "8"), now: 158))
    }

    func testTerminatedIDNeverUsedEvenWhenAnotherAppReusesIt() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 119))
        refresh(&lease, id: "2", at: 120)
        lease.terminated(id: "1") // It may now be another app's handle.
        XCTAssertFalse(lease.admitOpenRequest(request(ndp: "8"), now: 121))
        XCTAssertEqual(lease.openResponse(pending, ndi: "camera", now: 122), pending.response(ndi: "camera", accept: true, serviceHandle: "2"))
        XCTAssertEqual(lease.stop(now: 123), ["2"])
    }

    func testReusedOwnedIDDoesNotExtendOrReplaceOriginalAdmission() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 1))
        lease.terminated(id: "1")
        refresh(&lease, id: "1", at: 2)
        XCTAssertFalse(lease.admitOpenRequest(pending, now: 3))
        // Existing permission still refers to the exact originally admitted NDP.
        XCTAssertNotNil(lease.openResponse(pending, ndi: "camera", now: 4))
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 5))
    }

    func testNoLiveCurrentOrStoppedLeaseCannotAccept() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 1))
        lease.terminated(id: "1")
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 2))
        refresh(&lease, id: "2", at: 3)
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 4)) // Failure consumed permission.
        let next = request("2", ndp: "8")
        XCTAssertTrue(lease.admitOpenRequest(next, now: 5))
        XCTAssertEqual(lease.stop(now: 6), ["2"])
        XCTAssertNil(lease.openResponse(next, ndi: "camera", now: 7))
        XCTAssertFalse(lease.admitOpenRequest(next, now: 8))
        XCTAssertEqual(lease.stop(now: 9), [])
    }

    func testUnadmittedEncryptedWrongTupleAndExpiredRequestsFailClosed() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        XCTAssertFalse(lease.admitOpenRequest(request(csid: "1"), now: 1))
        XCTAssertFalse(lease.admitOpenRequest(request(csid: "invalid"), now: 1))
        let pending = request()
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 1))
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 2))
        XCTAssertNil(lease.openResponse(request(ndp: "8"), ndi: "camera", now: 3))
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 20)) // No deadline extension.
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 29)) // 3s command margin.
        XCTAssertFalse(lease.admitOpenRequest(pending, now: 33)) // Tombstone prevents revival.
    }

    func testNoResponseUsingLeaseNearExpiryAndNoUnboundedAdmissionGrowth() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 170))
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 178))
        XCTAssertFalse(lease.admitOpenRequest(request(ndp: "8"), now: 178))
        var bounded = NANPublicationLease(initialID: "1", started: 0)
        for id in 1...64 { XCTAssertTrue(bounded.admitOpenRequest(request(ndp: String(id)), now: 1)) }
        XCTAssertFalse(bounded.admitOpenRequest(request(ndp: "65"), now: 1))
        XCTAssertTrue(bounded.admitOpenRequest(request(ndp: "65"), now: 62))
    }

    func testRejectedPermissionDoesNotLeavePhantomOwnedNDP() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        var lifecycle = NDPRequestLifecycle()
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 1))
        XCTAssertEqual(lifecycle.request(pending, now: 1), [.accept(pending)])
        lease.terminated(id: "1")
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 2))
        lifecycle.responseNotAccepted(pending)
        XCTAssertEqual(lifecycle.stop(), [])
    }

    func testExplicitRejectionConsumesPermission() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 1))
        lease.finishRequest(pending)
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 2))
        XCTAssertEqual(lease.stop(now: 3), ["1"])
        XCTAssertEqual(lease.stop(now: 4), [])
    }

    func testUnsentRejectionCannotReleaseDifferentInitiatorTuple() {
        var lifecycle = NDPRequestLifecycle()
        let owned = request()
        _ = lifecycle.request(owned, now: 1)
        let different = NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=01:02:03:04:05:06 ndp_id=7 init_ndi=bb:bb:cc:dd:ee:ff publish_inst_id=1 csid=0", ownPublishID: "1")!
        lifecycle.responseNotAccepted(different)
        XCTAssertEqual(lifecycle.stop(), [.terminate(owned.ndp)])
    }

    func testKnownPendingRetransmissionDoesNotLosePermissionOnHandleReuse() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        var lifecycle = NDPRequestLifecycle()
        let prior = request(peer: "11:12:13:14:15:16", ndp: "4")
        _ = lifecycle.request(prior, now: 1)
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 2))
        XCTAssertEqual(lifecycle.request(pending, now: 2), [.terminate(prior.ndp)])
        lease.terminated(id: "1")
        refresh(&lease, id: "1", at: 3)
        XCTAssertFalse(lease.admitOpenRequest(pending, now: 4)) // Different generation.
        XCTAssertTrue(lifecycle.knowsRequest(pending))
        XCTAssertEqual(lifecycle.request(pending, now: 4), [])
        XCTAssertEqual(lifecycle.event(.disconnected(peerNMI: prior.peerNMI, id: prior.ndpID), now: 6), [.accept(pending)])
        XCTAssertNotNil(lease.openResponse(pending, ndi: "camera", now: 6))
    }

    func testCipherDifferenceOnKnownTupleCannotRejectOrReplaceOriginal() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        var lifecycle = NDPRequestLifecycle()
        let admitted = request()
        XCTAssertTrue(lease.admitOpenRequest(admitted, now: 1))
        _ = lifecycle.request(admitted, now: 1)
        XCTAssertNotNil(lease.openResponse(admitted, ndi: "camera", now: 1))
        let changed = request(csid: "1")
        XCTAssertTrue(lifecycle.knowsRequest(changed))
        XCTAssertEqual(lifecycle.expire(now: 2), []) // Production duplicate path.
        XCTAssertEqual(lifecycle.stop(), [.terminate(admitted.ndp)])

        var pendingLife = NDPRequestLifecycle()
        let prior = request(peer: "11:12:13:14:15:16", ndp: "4")
        _ = pendingLife.request(prior, now: 1)
        _ = pendingLife.request(admitted, now: 2)
        lease.terminated(id: "1")
        refresh(&lease, id: "1", at: 3)
        XCTAssertTrue(pendingLife.knowsRequest(changed))
        XCTAssertEqual(pendingLife.expire(now: 4), [])
        XCTAssertEqual(pendingLife.event(.disconnected(peerNMI: prior.peerNMI, id: prior.ndpID), now: 5), [.accept(admitted)])
        // The original open request was retained, never replaced by csid=1.
    }

    func testConsumedDuplicateDoesNotRejectAnAlreadyAcceptedHandshake() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        var lifecycle = NDPRequestLifecycle()
        let pending = request()
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 1))
        XCTAssertEqual(lifecycle.request(pending, now: 1), [.accept(pending)])
        XCTAssertNotNil(lease.openResponse(pending, ndi: "camera", now: 1))
        XCTAssertTrue(lease.admitOpenRequest(pending, now: 2))
        XCTAssertEqual(lifecycle.request(pending, now: 2), [])
        XCTAssertNil(lease.openResponse(pending, ndi: "camera", now: 2))
    }
}
