# Raspberry Pi NAN camera

This Swift Wendy app publishes `wendy.nan.camera.v1` directly through its
`nan` entitlement. It accepts an open NAN data path from the companion Pixel
app and serves JPEG snapshots and saved photos over TCP port 9091. The complete
wire contract is [../PROTOCOL.md](../PROTOCOL.md).

The Wendy agent provisions a dedicated NAN data interface and the host-visible
control client directory. The app owns the publish handle and accepts its own
NDP requests. The TCP listener binds only the interface's scoped IPv6
link-local address. `network: host` is required for that link-local socket;
it does not publish the service on Ethernet or infrastructure Wi-Fi.
The app publishes and starts its NDP responder before waiting for the NDI's
link-local address: the interface has no such address until a peer connects.
SIGTERM interrupts that wait and withdraws the publish handle.

Discovery uses a 180-second supplicant lease. Every 120 seconds the app first
creates a replacement publication, accepts requests on both advertised IDs,
and cancels the old ID after 35 seconds. An admitted open request has one
bounded response authorization tied to its original publication generation and
exact peer/NDP tuple. If a deferred handoff outlives the old publication, its
response uses the current owned live handle for the same fixed open camera
service. The original request and NDP ID remain unchanged. This is specific to
the demo's immutable open service; it must not be generalized to distinct
services or security contexts. Missing/expired authorization or a missing live
handle causes rejection, and duplicate requests never extend the deadline.
Established NDPs and TCP sockets are independent of the discovery handle.
A full service table retains the old lease and retries once per five seconds;
only two camera publications overlap, and no extra data interface is needed.
SIGTERM cancels the app's currently owned, unexpired handles. Expired or
terminated IDs are discarded, and ambiguous cancellation commands are never
retried because another app may reuse the ID.

After SIGKILL, discovery expires within 180 seconds of its last publication,
plus the next active NAN discovery window. If the radio is suspended, expiry
is checked when discovery resumes. This bounds orphan advertisements; it does
not claim that SIGKILL gracefully terminates NDPs. The direct socket entitlement
is trusted and does not supervise arbitrary apps' publish or NDP ownership.

Android can reuse its NDI MAC after randomizing its NAN management address.
Before accepting that replacement, the app terminates only its own old NDPs
for the same peer NDI and waits for their disconnect confirmations. The wait
is bounded to six seconds per queued handle on the busiest old peer, capped
at twenty seconds (hostap stale-peer termination takes about four seconds).
Terminations are serialized per peer management address, even when several
peer data interfaces are being replaced. A timeout rejects the new request
instead of reusing an obsolete station. Failed setup disconnects with an
unassigned local NDI release only exact already-owned or pending requests.
Other apps’ and mesh sessions are never inferred to be owned.

`camera` grants access to `/dev/video*`. The app keeps a 320×240 V4L2 capture
stream open between snapshot requests. A still briefly switches to 1920×1080,
then the next snapshot reopens the low-resolution stream. UVC MJPEG is passed
through without transcoding; YUYV devices use libjpeg-turbo's default encoding
settings. The app does not expose or override JPEG quality. The camera may
negotiate a different resolution if a requested mode is unavailable. Thumbnails
are encoded from the saved JPEG. `WENDY_CAMERA_DEVICE=/dev/videoN` overrides
automatic capture-device selection.

The `persist` entitlement stores at most 100 photos and 512 MiB of app-owned
originals plus thumbnails in `/photos`. A save reserves both sizes before writing;
startup and subsequent saves remove owned temporary files and UUID-named orphan
thumbnails left by a failed
commit or deletion. Failed orphan cleanup blocks further saves, and incomplete
thumbnail deletion is reported as an error that can be retried. Valid originals
are preserved during recovery.
Images are named by Pi-generated UUIDs; a wire ID never becomes an arbitrary
path. Incomplete temporary writes are cleaned on startup. The Android app
streams downloads from this app in 64 KiB writes, allowing rate updates as the
bytes arrive.

## Build and run

```sh
docker buildx build --platform linux/arm64 --load -t wendy-rpi-nan-camera:dev .
wendy run --device 192.168.2.5 --builder docker --detach --yes
```

The Docker build runs the protocol and storage tests. It does not need a
webcam. Deployment requires a WendyOS agent with the new `nan` entitlement and
an installed BE202 driver. Radio and webcam acceptance must be run on the Pi5
with the Pixel client after deployment.

### Control command timeouts

The supplicant command channel uses unnumbered Unix datagrams. After an ambiguous
response failure the camera retires that endpoint and opens a fresh pathname for
the next command, so late replies cannot be attributed to another operation.
Commands are not automatically replayed. Both send and receive waits are bounded;
ordinary event-listener receive timeouts retain the separate ATTACH endpoint.

A lost `NAN_PUBLISH` response may leave an unknown finite 180-second publication.
The camera cannot safely guess or cancel that ID. Initial publication keeps the
process alive through a 183-second, stop-cancelable cooldown before another
attempt (including the final attempt before returning an error), avoiding an
immediate replay through the container restart policy. Periodic renewal records
the same cooldown without sleeping the event loop or interrupting established
TCP/NDP sessions. Definitive FAIL responses retain the existing bounded retry
policy. This cooldown is measured from the timeout; it is not evidence of the
unknown publication's creation time or its actual removal. Hostap's finite lease
still governs an orphan's expiry, and no other app's IDs are canceled.
