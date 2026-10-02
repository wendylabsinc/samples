import XCTest
@testable import CameraWire

final class NANPublicationLeaseTests: XCTestCase {
    func testRefreshAcceptsBothPublicationsUntilRetirement() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        XCTAssertNil(lease.beginRefresh(now: 119))
        let token = lease.beginRefresh(now: 120)!
        XCTAssertNil(lease.beginRefresh(now: 121))
        XCTAssertEqual(lease.acceptingIDs(now: 121), ["1"])
        XCTAssertEqual(lease.completeRefresh(token: token, id: "2", now: 122), [])
        XCTAssertEqual(lease.acceptingIDs(now: 131), ["1", "2"])
        XCTAssertEqual(lease.acceptingIDs(now: 132), ["1", "2"])
        XCTAssertEqual(lease.cancellations(now: 156), [])
        XCTAssertEqual(lease.cancellations(now: 157), ["1"])
        XCTAssertEqual(lease.cancellations(now: 158), [])
        XCTAssertEqual(lease.stop(now: 159), ["2"])
    }

    func testCapacityFailureKeepsOldPublicationAndBoundsRetries() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let first = lease.beginRefresh(now: 120)!
        XCTAssertEqual(lease.completeRefresh(token: first, id: nil, now: 123), [])
        XCTAssertEqual(lease.acceptingIDs(now: 127), ["1"])
        XCTAssertNil(lease.beginRefresh(now: 127))
        let retry = lease.beginRefresh(now: 128)!
        XCTAssertEqual(lease.completeRefresh(token: retry, id: "2", now: 129), [])
        XCTAssertEqual(lease.stop(now: 130), ["1", "2"])
        XCTAssertEqual(lease.stop(now: 131), [])
        XCTAssertNil(lease.beginRefresh(now: 999))
    }

    func testExpiredAndTerminatedHandlesAreNeverCanceledAfterReuse() {
        var expired = NANPublicationLease(initialID: "1", started: 0)
        XCTAssertEqual(expired.acceptingIDs(now: 180), [])
        XCTAssertEqual(expired.stop(now: 181), [])
        var terminated = NANPublicationLease(initialID: "2", started: 0)
        terminated.terminated(id: "2")
        // Another app can now own 2. Only a newly returned handle is ours.
        let token = terminated.beginRefresh(now: 1)!
        XCTAssertEqual(terminated.completeRefresh(token: token, id: "3", now: 2), [])
        XCTAssertEqual(terminated.stop(now: 3), ["3"])
        var margin = NANPublicationLease(initialID: "4", started: 0)
        XCTAssertEqual(margin.stop(now: 178), [])
    }

    func testStopDuringRefreshCancelsLateSuccessExactlyOnce() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let token = lease.beginRefresh(now: 120)!
        XCTAssertEqual(lease.stop(now: 121), ["1"])
        XCTAssertEqual(lease.completeRefresh(token: token, id: "2", now: 122), ["2"])
        XCTAssertEqual(lease.completeRefresh(token: token, id: "2", now: 123), [])
        XCTAssertEqual(lease.stop(now: 124), [])
    }

    func testDelayedCompletionAndDelayedCancelCannotTouchReusedHandles() {
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let token = lease.beginRefresh(now: 120)!
        XCTAssertEqual(lease.completeRefresh(token: token, id: "2", now: 121), [])
        XCTAssertEqual(lease.cancellations(now: 179), [])
        XCTAssertEqual(lease.stop(now: 181), ["2"])
        var delayed = NANPublicationLease()
        let pending = delayed.beginRefresh(now: 0)!
        XCTAssertEqual(delayed.completeRefresh(token: pending, id: "9", now: 181), [])
        XCTAssertEqual(delayed.stop(now: 182), [])
    }

    func testPublicationReplacementPreservesIndependentOwnedNDP() {
        var paths = NDPRequestLifecycle()
        let request = NDPRequest(event: "NAN-NDP-REQUEST peer_nmi=01:02:03:04:05:06 ndp_id=1 init_ndi=aa:bb:cc:dd:ee:ff publish_inst_id=1", ownPublishID: "1")!
        XCTAssertEqual(paths.request(request, now: 0), [.accept(request)])
        XCTAssertEqual(paths.event(.connected(request.ndp), now: 0), [])
        var lease = NANPublicationLease(initialID: "1", started: 0)
        let token = lease.beginRefresh(now: 120)!
        _ = lease.completeRefresh(token: token, id: "2", now: 121)
        XCTAssertEqual(lease.cancellations(now: 156), ["1"])
        XCTAssertEqual(paths.expire(now: 156), [])
        XCTAssertEqual(paths.stop(), [.terminate(request.ndp)])
    }
}
