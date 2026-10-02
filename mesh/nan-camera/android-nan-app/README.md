# Wendy NAN Camera for Pixel 7

This Android app passively subscribes to the Raspberry Pi camera's `wendy.nan.camera.v1`
Wi-Fi Aware service and opens an unencrypted NAN data path. It implements the
framed TCP protocol in [../PROTOCOL.md](../PROTOCOL.md). Discovery and the NDP
are owned by this app, separately from Wendy's mesh.

The dark interface adapts to portrait and landscape with system-bar insets.
The camera session survives rotation; a compact landscape view keeps the whole
preview and capture action visible. Live/recent download speed stays visible
while browsing the gallery.

The app shows successive 320×240 snapshots while connected. It can capture a
photo, list saved photos, display a thumbnail, delete a photo, and stream a full
JPEG to `Pictures/Wendy NAN Camera`. During a download it pauses snapshots and
displays a live transfer rate. The final average remains visible for ten seconds.
Downloads allocate one buffer, bounded to 32 MiB, before sending the request.
Timing starts immediately after the request flush returns and ends when the full
JPEG has been received in that RAM buffer. The existing connection has no other
outstanding application request. Android destination setup and saving are outside
this measurement; reply latency remains included. A flush confirms handing the
request to TCP, not its physical transmission over the radio.

Build and test with the Android SDK installed:

```sh
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' \
ANDROID_HOME="$HOME/Library/Android/sdk" \
./gradlew testDebugUnitTest assembleDebug
```

The APK is `app/build/outputs/apk/debug/app-debug.apk`. Android 13 or later is
required. Grant the Nearby devices permission on the first discovery. The app
uses `Network.socketFactory`, so a concurrent infrastructure Wi-Fi connection
is not rebound to the NAN data path.

Initial TCP establishment allows five seconds per socket attempt within a ten-second total deadline, accommodating NAN availability windows and IPv6 neighbor setup. Immediate refusals retry promptly; canceling the connection closes the pending socket.

Incoming preview and thumbnail JPEGs are inspected before pixel allocation. Each
dimension must be at most 4096 pixels and the decoded image at most 4,194,304
pixels (16 MiB for the requested ARGB8888 buffer). This is separate from the
32 MiB compressed wire limit; oversized or non-JPEG replies disconnect cleanly.
