# Pi NAN camera protocol v1

The Pi publishes the Wi-Fi Aware service `wendy.nan.camera.v1` with synchronized discovery and an open NAN data path. This v1 service uses fixed TCP port 9091 and omits service-specific information: physical Pixel 7 testing found that its firmware ignored hostap discovery frames carrying nonempty service info, while the same service with no info and `data_path=1` was discovered. The Android client subscribes to that service, requests an open data path, then opens TCP port 9091 through the resulting Android `Network` to the peer IPv6 link-local address. The Pi server binds only its dedicated NAN data interface, not Ethernet or infrastructure Wi-Fi. The TCP frame header checks protocol version 1.

Every TCP message has a 14-byte header:

| Offset | Size | Field |
| --- | ---: | --- |
| 0 | 4 | ASCII magic `WNCA` |
| 4 | 1 | Version `1` |
| 5 | 1 | Operation |
| 6 | 4 | Request ID, unsigned big-endian |
| 10 | 4 | Payload length, unsigned big-endian |

The payload follows immediately. Maximum payload is 32 MiB; both sides reject a larger length before allocation. A reply echoes the request ID and sets the high bit of the request operation. Operation `0x7f` is an error reply with a bounded UTF-8 message. Requests are serialized on one connection; the client can reconnect after a disconnect.

A complete, valid error reply leaves the stream usable for another request. For example, an unplugged webcam or full photo store does not require a new NAN data path. The Android client reports that operation's error and waits briefly before retrying live view. If a reply fails after any bytes may have been sent, the server closes the connection without appending an error frame to the incomplete payload. The client must reconnect after a truncated reply or malformed frame.

| Request op | Request payload | Successful reply payload |
| --- | --- | --- |
| `0x01` snapshot | empty | Current low-resolution JPEG, target 320×240 |
| `0x02` take photo | empty | One 32-byte photo record |
| `0x03` list photos | empty | `count:u16` then that many 32-byte photo records |
| `0x04` thumbnail | 16-byte photo UUID | Thumbnail JPEG, target 160×120 |
| `0x05` download | 16-byte photo UUID | Full-resolution JPEG, streamed in chunks after the header |
| `0x06` delete | 16-byte photo UUID | empty |

A photo record is `uuid:16` in standard UUID byte order, `capturedAtUnixMs:u64` big-endian, and `jpegBytes:u64` big-endian. IDs originate on the Pi and are never treated as file paths from the wire. A photo save writes a temporary file and renames it only after the full JPEG is present. The server bounds its saved-photo count and total bytes.

The Android app repeatedly requests snapshots while connected and no download is active. During a download it stops snapshot requests, streams the reply body to a pending MediaStore item, measures bytes per monotonic elapsed time, shows the current transfer rate, and keeps the final rate visible briefly after completion.

The Pi app owns publish, NDP response and terminate operations through its `nan` entitlement. The entitlement provides `WENDY_NAN_SOCKET`, `WENDY_NAN_NDI`, and `WENDY_NAN_CLIENT_DIR`. Its Unix datagram control client **must bind its local pathname inside `WENDY_NAN_CLIENT_DIR`** so wpa_supplicant can reply across the container mount boundary. `network: host` is a separate entitlement for the NDP data socket. The physical radio may share one NAN cluster with mesh activity; this app's service and sessions are distinct.
