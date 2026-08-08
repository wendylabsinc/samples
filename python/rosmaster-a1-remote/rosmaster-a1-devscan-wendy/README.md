# Rosmaster A1 Devscan

Standalone on-demand diagnostic tool for serial device discovery. Prints a census of `/dev/serial/by-id` symlinks, ttyUSB and ttyACM node listings, and sysfs tty inventory to identify which serial adapters the host actually has.

Deliberately declares **no serial entitlements**—it requests only network (host mode) and USB access. This ensures deployment always succeeds, even when named serial devices are missing or renumbered; the absence of a device is not an entitlement failure, it is the diagnosis.

**Not part of the multi-service `rosmaster-a1` app.** Deploy on demand:

```bash
cd rosmaster-a1-devscan-wendy && wendy run --yes --device <car-hostname>.local:50052
```

Output from the entrypoint is prefixed with `DEVSCAN` for easy grepping.
