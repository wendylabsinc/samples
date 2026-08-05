"""Reset the USB adapter behind a serial port, for when it wedges.

The LiDAR's CH340 enumerates and its /dev node exists, but nothing can open it:
the YDLIDAR driver reports "cannot bind to the specified serial port", and a
read only census of the same port hangs rather than returning an error. A node
that exists yet cannot be opened by anyone is the signature of a wedged USB
device rather than a configuration problem, and the usual remedy is to make the
kernel re-enumerate it.

This walks from the tty node to its USB parent and issues USBDEVFS_RESET, the
same technique the HP60C supervisor already uses when the depth camera stops
producing frames. It is a long shot for a device that is simply unplugged, and
it says so in the log rather than pretending otherwise.

Usage: python3 usb_reset.py /dev/ttyUSB1
"""

import fcntl
import json
import os
import sys
import time
from pathlib import Path

USBDEVFS_RESET = 21780  # _IO('U', 20)


def log(**fields):
    print("LIDAR_USB_RESET " + json.dumps(fields, sort_keys=True, default=str), flush=True)


def usb_parent(tty_path):
    """Walk from /dev/ttyUSBn up to the USB device directory that owns it."""
    name = os.path.basename(os.path.realpath(tty_path))
    start = Path("/sys/class/tty") / name / "device"
    if not start.exists():
        return None
    node = start.resolve()
    # The tty hangs off a USB interface; busnum and devnum live on the device
    # a level or two above it.
    for _ in range(6):
        if (node / "busnum").exists() and (node / "devnum").exists():
            return node
        if node.parent == node:
            break
        node = node.parent
    return None


def main():
    tty = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"
    if not os.path.exists(tty):
        log(event="skip", tty=tty, reason="device node does not exist")
        return 0

    parent = usb_parent(tty)
    if parent is None:
        log(event="skip", tty=tty, reason="no USB parent with busnum and devnum found")
        return 0

    try:
        bus = int((parent / "busnum").read_text().strip())
        dev = int((parent / "devnum").read_text().strip())
    except (OSError, ValueError) as exc:
        log(event="skip", tty=tty, reason="could not read busnum or devnum", detail=str(exc))
        return 0

    node = Path("/dev/bus/usb/%03d/%03d" % (bus, dev))
    ident = {}
    for field in ("idVendor", "idProduct", "product", "manufacturer"):
        try:
            ident[field] = (parent / field).read_text().strip()
        except OSError:
            pass
    log(event="target", tty=tty, usb_node=str(node), sysfs=str(parent), **ident)

    if not node.exists():
        log(event="skip", reason="usb node missing, the usb entitlement may not be granted")
        return 0

    try:
        with node.open("wb", buffering=0) as handle:
            fcntl.ioctl(handle, USBDEVFS_RESET, 0)
        log(event="reset_sent", usb_node=str(node))
    except OSError as exc:
        log(event="reset_failed", usb_node=str(node), errno=exc.errno, detail=str(exc)[:160])
        return 0

    # Re-enumeration takes a moment and the tty node may briefly disappear.
    for _ in range(20):
        time.sleep(0.5)
        if os.path.exists(tty):
            break
    log(event="after_reset", tty=tty, exists=os.path.exists(tty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
