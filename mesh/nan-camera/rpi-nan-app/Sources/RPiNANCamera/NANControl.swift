import Foundation
import Glibc
import CameraC
import CameraWire

final class NANPublisher: @unchecked Sendable {
    private let commands: WPASocket
    private let events: WPASocket
    private let ndi: String
    private let ndiMAC: String
    private var publication: NANPublicationLease
    private let lifecycleLock = NSLock()
    private var lifecycle = NDPRequestLifecycle()
    private var stopped = false

    init() throws {
        let env = ProcessInfo.processInfo.environment
        guard let socket = env["WENDY_NAN_SOCKET"], let ndi = env["WENDY_NAN_NDI"],
              let clientDir = env["WENDY_NAN_CLIENT_DIR"] else {
            throw NANError.failed("nan entitlement environment missing")
        }
        commands = try WPASocket(remotePath: socket, clientDirectory: clientDir, receiveTimeoutSeconds: 3)
        events = try WPASocket(remotePath: socket, clientDirectory: clientDir, receiveTimeoutSeconds: 1)
        self.ndi = ndi
        ndiMAC = try String(contentsOfFile: "/sys/class/net/\(ndi)/address", encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard try events.command("ATTACH") == "OK" else { throw NANError.failed("NAN event attach rejected") }
        _ = try commands.command("NAN_SCHED_CONFIG_MAP map_id=1 2437:0e000000")
        // Pixel 7 ignores nonempty SSI, so the service name fixes the port;
        // the framed TCP header carries the version after NDP establishment.
        let control = commands
        let (published, publishedAt) = try initialNANPublication {
            try control.command(Self.publishCommand)
        }
        guard let numeric = UInt32(published), numeric > 0 else { throw NANError.failed("invalid publish handle") }
        publication = NANPublicationLease(initialID: published, started: publishedAt)
        print("[nan-camera] published wendy.nan.camera.v1 handle=\(published) ndi=\(ndi)")
    }

    private static let publishCommand = "NAN_PUBLISH service_name=wendy.nan.camera.v1 sync=1 data_path=1 solicited=0 ttl=180"

    deinit { stop() }

    private func cancelPublications(_ ids: [String]) {
        for id in ids {
            // The lease has already relinquished this handle. Do not retry an
            // ambiguous cancellation: another app could reuse the freed ID.
            do {
                _ = try commands.command("NAN_CANCEL_PUBLISH publish_id=\(id)")
                print("[nan-camera] canceled discovery lease handle=\(id)")
            } catch {
                print("[nan-camera] discovery cancellation not retried handle=\(id): \(error)")
            }
        }
    }

    private func refreshPublication(now: TimeInterval) {
        cancelPublications(publication.cancellations(now: now))
        guard let token = publication.beginRefresh(now: now) else { return }
        do {
            let id = try commands.command(Self.publishCommand)
            guard let numeric = UInt32(id), numeric > 0 else { throw NANError.failed("invalid publish handle") }
            cancelPublications(publication.completeRefresh(token: token, id: id,
                               now: ProcessInfo.processInfo.systemUptime))
            print("[nan-camera] renewed discovery lease handle=\(id) ttl=180")
        } catch {
            _ = publication.completeRefresh(token: token, id: nil,
                        now: ProcessInfo.processInfo.systemUptime,
                        outcomeUnknown: (error as? NANError)?.outcomeUnknown == true)
            print("[nan-camera] discovery lease refresh deferred: \(error)")
        }
    }

    func stop() {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        if stopped { return }
        stopped = true
        perform(lifecycle.stop())
        cancelPublications(publication.stop(now: ProcessInfo.processInfo.systemUptime))
    }

    /// Called under lifecycleLock so stop cannot interleave with an ACCEPT.
    /// Commands are short control transactions; we never wait here for an NDP
    /// disconnect. That confirmation is consumed by the next event-loop turn.
    private func perform(_ actions: [NDPAction]) {
        var queue = actions
        var index = 0
        while index < queue.count {
            let action = queue[index]
            var performedAction = action
            index += 1
            let command: String
            switch action {
            case .accept(let request):
                if let response = publication.openResponse(request, ndi: ndi,
                        now: ProcessInfo.processInfo.systemUptime) {
                    command = response
                } else {
                    lifecycle.responseNotAccepted(request)
                    performedAction = .reject(request)
                    command = request.response(ndi: ndi, accept: false)
                    print("[nan-camera] open request authorization expired or no live service handle")
                }
            case .reject(let request):
                publication.finishRequest(request)
                command = request.response(ndi: ndi, accept: false)
            case .terminate(let ndp):
                command = ndp.terminateCommand
            }
            do {
                _ = try commands.command(command)
                print("[nan-camera] \(command)")
            } catch {
                print("[nan-camera] NDP command error: \(command): \(error)")
                queue += lifecycle.commandFailed(performedAction)
            }
        }
    }

    private func handle(_ raw: String?) {
        lifecycleLock.lock()
        defer { lifecycleLock.unlock() }
        guard !stopped, wendy_stop_requested() == 0 else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if let raw, raw.contains("NAN-PUBLISH-TERMINATED"),
           let field = raw.split(separator: " ").first(where: { $0.hasPrefix("publish_id=") }) {
            publication.terminated(id: String(field.dropFirst("publish_id=".count)))
            print("[nan-camera] \(raw)")
        }
        if let raw, let request = publication.acceptingIDs(now: now).compactMap({ NDPRequest(event: raw, ownPublishID: $0) }).first {
            if lifecycle.knowsRequest(request) {
                // Preserve the original admitted request/security context. A
                // duplicate cannot replace it or reject its in-flight tuple.
                perform(lifecycle.expire(now: now))
            } else if publication.admitOpenRequest(request, now: now) {
                perform(lifecycle.request(request, now: now))
            } else {
                perform([.reject(request)])
            }
        } else if let raw, raw.contains("NAN-NDP-CONNECTED") || raw.contains("NAN-NDP-DISCONNECTED") {
            print("[nan-camera] \(raw)")
            perform(lifecycle.event(raw: raw, localNDIMAC: ndiMAC, now: now))
        } else {
            perform(lifecycle.expire(now: now))
        }
        // Refresh discovery only; established NDPs and TCP sockets retain their
        // independent ownership and survive old-publication cancellation.
        refreshPublication(now: ProcessInfo.processInfo.systemUptime)
    }

    func runEvents() {
        while wendy_stop_requested() == 0 {
            do {
                handle(try events.receive())
            } catch {
                // The one-second receive timeout also drives pending handoff
                // expiry when no further supplicant events arrive.
                handle(nil)
                if wendy_stop_requested() == 0, !String(describing: error).contains("wpa recv: 11") {
                    print("[nan-camera] NAN event error: \(error)")
                    Thread.sleep(forTimeInterval: 0.2)
                }
            }
        }
    }
}
