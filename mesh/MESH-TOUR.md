# Mesh tour

Use two devices enrolled in the same organisation, a mesh-capable WendyOS
agent and CLI, and Docker. Run these commands from `samples/mesh`.

## Enable mesh and internet sharing

```bash
A=192.168.2.2   # First device; replace with your endpoint.
B=192.168.2.3   # Second device.
for DEVICE in "$A" "$B"; do
  wendy --device "$DEVICE" device local-mesh configure \
    --participate=true --ble=true --ethernet=true --infrastructure-wifi=true
  wendy --device "$DEVICE" device local-mesh status
done
```

Choose carriers your devices support; add `--nan=true` for Wi-Fi Aware.
Look for authenticated peers in `status`.

```bash
# A offers its internet connection; B uses mesh internet when it has no uplink.
wendy --device "$A" device local-mesh configure --share-uplink=true
wendy --device "$B" device local-mesh configure --roam=true
```

Omitted options keep their saved values; use `=false` to disable an option.

### Try roaming without a local uplink

Keep A online and sharing, with an authenticated BLE or NAN link to B.
Enable roaming on B, disconnect its Wi-Fi, then unplug its Ethernet cable:

```bash
wendy --device "$B" device local-mesh configure --roam=true
wendy --device "$B" device wifi disconnect
# Now unplug B's Ethernet cable; keep A's internet connection intact.
B_CLOUD="tom-rpi-5"   # Replace with B's enrolled cloud name.
wendy --device "$B_CLOUD" --json cloud device local-mesh status
```

With the CLI logged into the same organisation, the cloud command still reaches
B through its shared internet connection. Look for the donor's `gatewayAssetId`,
an authenticated peer, and `roaming (using signed mesh gateway)` in `detail`.
The local hostname may no longer resolve; `cloud device` uses the tunnel broker.

## time-publisher: JSON over HTTP

Publishes `_wendytime._tcp` using raw mDNS, without Avahi. Serves the current
UTC time and hostname as JSON on port 8080.

```bash
wendy --device "$A" run --prefix time-publisher --dockerfile Dockerfile --builder docker --detach --yes --skip-cloud-registration
wendy --device "$A" device logs --app com.example.go-time-publisher --tail 20 --no-follow
```

Expect `published ...` and `serving http://0.0.0.0:8080/`.

## time-browser: discovery and polling

Discovers publishers and polls them every 10 seconds. Logs show `discovered`,
then `addr=10.99.x.x:8080 status=200 answer={...}` for a mesh peer.
Stopping a publisher produces `FAILED` polls, then `lost` after record expiry.

```bash
wendy --device "$B" run --prefix time-browser --dockerfile Dockerfile --builder docker --detach --yes --skip-cloud-registration
wendy --device "$B" device logs --app com.example.go-time-browser --tail 40 --no-follow
```

## time-web: HTML through Avahi

Serves an automatically refreshing time page on port 8081. Publishes
`_http._tcp` through the Avahi/DNS-SD API; the agent supplies its private
Avahi daemon and D-Bus via the app's `avahi` entitlement.

```bash
wendy --device "$A" run --prefix time-web --dockerfile Dockerfile --builder docker --detach --yes --skip-cloud-registration
wendy --device "$A" device logs --app com.example.go-time-web --tail 20 --no-follow
# Switch the browser to the web sample.
wendy --device "$B" run --prefix time-browser --dockerfile Dockerfile --builder docker --detach --yes --skip-cloud-registration --env TIME_BROWSER_SERVICE_TYPE=_http._tcp
wendy --device "$B" device logs --app com.example.go-time-browser --tail 40 --no-follow
```

Expect `avahi-compat service registered` on A, then discovery and
`addr=10.99.x.x:8081 status=200` with HTML on B.

## NAN camera pair: customer controller connectivity

[nan-camera/rpi-nan-app](nan-camera/rpi-nan-app) captures and stores photos;
[nan-camera/android-nan-app](nan-camera/android-nan-app) previews, captures and
downloads them on Android. This demonstrates a customer's controller connecting
directly to a device over Wi-Fi Aware (NAN). The Pi uses Wendy's `nan`
entitlement, but discovery and data transfer belong to the apps, without Wendy
mesh routing or mesh internet sharing. The current demo uses an open,
unencrypted NAN data path.

Requires a Pi with a supported NAN adapter and camera, and an Android 13+
phone supporting Wi-Fi Aware.

```bash
PI=192.168.2.5
wendy --device "$PI" run --prefix nan-camera/rpi-nan-app --dockerfile Dockerfile --builder docker --detach --yes --skip-cloud-registration
wendy --device "$PI" device logs --app sh.wendy.nan.rpi-camera --tail 20 --no-follow
(cd nan-camera/android-nan-app && ./gradlew assembleDebug)
adb install -r nan-camera/android-nan-app/app/build/outputs/apk/debug/app-debug.apk
```

See the Android [README](nan-camera/android-nan-app/README.md) for SDK/JDK setup.
Open **Wendy NAN Camera**, grant Nearby devices permission, scan and connect.
Expect a live preview; use **Take photo** and **View** to download the original.
Preview requests pause during downloads; the footer shows the transfer speed.
