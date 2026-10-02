# Direct NAN camera example

A Raspberry Pi camera app and Android customer controller connect directly
using Wi-Fi Aware (NAN). The Pi uses Wendy's `nan`, `camera` and `persist`
entitlements. The apps own discovery and their TCP photo protocol; they do
not use Wendy mesh routes or mesh internet sharing. The demo data path is
open and unencrypted.

- [Pi camera](rpi-nan-app/README.md): live previews, capture, stored originals.
- [Android controller](android-nan-app/README.md): discovery, preview, gallery,
  full-resolution downloads and transfer-rate display.
- [Protocol](PROTOCOL.md): shared framing and commands.
- `entitlement-smoke`: a small direct-NAN control-socket check used during
  platform validation; not needed to run the camera demo.

For a quick start, see [the mesh tour](../MESH-TOUR.md#nan-camera-pair-customer-controller-connectivity).
