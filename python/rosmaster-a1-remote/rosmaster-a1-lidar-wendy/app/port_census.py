"""Report what every serial port on the car actually is.

The LiDAR driver fails with "cannot bind to the specified serial port", which
is ambiguous: the port may be missing, permission denied, held by another
process, or present and simply silent because nothing is plugged into it. Those
need different fixes and the driver's message does not distinguish them.

For each candidate port this prints one PORT_CENSUS line covering: whether the
node exists, its permissions and owner, whether it can be opened, the errno if
not, and how many bytes arrive within a short listen window at each baud rate
the hardware might use. A port that opens but stays silent at every baud is
almost certainly nothing plugged in, which is a cable to chase rather than a
config to change.

Read only: it opens ports and listens, and never writes.
"""

import glob
import json
import os
import signal
import stat
import sys
import time

BAUDS = [int(b) for b in os.environ.get("PORT_CENSUS_BAUDS", "230400,512000,115200").split(",")]
LISTEN_S = float(os.environ.get("PORT_CENSUS_LISTEN_S", "1.0"))
OPEN_TIMEOUT_S = float(os.environ.get("PORT_CENSUS_OPEN_TIMEOUT_S", "5.0"))


def log(**fields):
    sys.stderr.write("PORT_CENSUS " + json.dumps(fields, sort_keys=True, default=str) + "\n")
    sys.stderr.flush()


def describe(path):
    info = {"port": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    try:
        st = os.stat(path)
        info["mode"] = stat.filemode(st.st_mode)
        info["uid"] = st.st_uid
        info["gid"] = st.st_gid
        info["is_char_device"] = stat.S_ISCHR(st.st_mode)
    except OSError as exc:
        info["stat_error"] = str(exc)
    info["readable"] = os.access(path, os.R_OK)
    info["writable"] = os.access(path, os.W_OK)
    try:
        info["realpath"] = os.path.realpath(path)
    except OSError:
        pass
    return info


def deadline(seconds, label):
    """Arm SIGALRM so a blocking syscall cannot wedge the census.

    O_NONBLOCK is meant to make open() return immediately, but a wedged USB
    serial driver can still block inside the kernel. An earlier version of this
    census stopped dead at that point and printed nothing for the port, which
    told us nothing at all. Every syscall that touches a device now runs under
    a timer, so the worst case is a line saying it timed out.
    """
    def fired(signum, frame):
        raise TimeoutError("%s exceeded %ss" % (label, seconds))

    previous = signal.signal(signal.SIGALRM, fired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def disarm(previous):
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def raw_open(path):
    """Open the node without blocking, and report the errno if that fails.

    A tty whose device is absent or unpowered can block indefinitely on open
    while the driver waits for carrier. The first version of this census did
    exactly that on the LiDAR port and simply produced no line for it, which is
    the least useful possible outcome. O_NONBLOCK turns that hang into an
    immediate answer, and the errno distinguishes "busy" from "denied" from
    "no such device".
    """
    previous = deadline(OPEN_TIMEOUT_S, "open(%s)" % path)
    fd = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        return {"raw_open": True}
    except TimeoutError as exc:
        return {"raw_open": False, "errno": None, "strerror": "timeout",
                "detail": str(exc)[:160]}
    except OSError as exc:
        return {"raw_open": False, "errno": exc.errno,
                "strerror": exc.strerror, "detail": str(exc)[:160]}
    finally:
        disarm(previous)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def listen(path, baud):
    """Open at one baud rate and count bytes seen. Returns a result dict."""
    import serial

    # A blocking open would hang the whole census, so cap it. SIGALRM is enough
    # here: this runs single threaded, before anything else has started.
    previous = deadline(OPEN_TIMEOUT_S + LISTEN_S, "listen(%s@%s)" % (path, baud))
    try:
        with serial.Serial(path, baud, timeout=0.2) as ser:
            stop_at = time.time() + LISTEN_S
            seen = b""
            while time.time() < stop_at:
                chunk = ser.read(256)
                if chunk:
                    seen += chunk
                    if len(seen) >= 512:
                        break
            return {"baud": baud, "opened": True, "bytes": len(seen),
                    "head": seen[:16].hex() if seen else ""}
    except Exception as exc:  # noqa: BLE001 - reporting the failure is the point
        return {"baud": baud, "opened": False,
                "error": type(exc).__name__, "detail": str(exc)[:160],
                "errno": getattr(exc, "errno", None)}
    finally:
        disarm(previous)


def main():
    candidates = sorted(set(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")))
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    log(event="scan", tty_nodes=candidates, by_id=by_id)

    for path in candidates:
        info = describe(path)
        info.update(raw_open(path))
        results = []
        if info.get("exists") and info.get("raw_open"):
            for baud in BAUDS:
                res = listen(path, baud)
                results.append(res)
                # A port that will not open at one baud will not open at another,
                # so stop rather than repeating an identical permission error.
                if not res["opened"]:
                    break
                if res["bytes"]:
                    break
        info["listen"] = results
        talkative = any(r.get("bytes") for r in results)
        blocked = (not info.get("raw_open")) or (results and not results[0]["opened"])
        info["verdict"] = ("blocked" if blocked
                           else "traffic" if talkative
                           else "silent")
        log(event="port", **info)

    log(event="done", hint="silent means the port opens but nothing is transmitting: "
                           "suspect an unplugged or unpowered device rather than config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
