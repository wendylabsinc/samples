"""Identify which serial adapter is the Rosmaster motor board.

The car carries two CH340 USB serial adapters, one for the motor board and one
for the LiDAR. They enumerate in whatever order the kernel finds them, so
ttyUSB0 and ttyUSB1 are not stable across reboots or re-plugs. Opening the
wrong one succeeds: the port opens, the bridge reports "connected", and every
drive command is written into a device that ignores it. That failure is silent
and looks identical to working hardware from the outside.

So probe instead of guessing. Open each candidate, ask the board for its
firmware version, and treat a plausible answer as proof. Every attempt prints a
SERIAL_IDENTIFY line to stderr so the choice can be reconstructed from logs
without anyone watching the wheels. The winning port goes to stdout, which is
the only thing on stdout, so the caller can capture it directly.

The winning port is written to the file named by SERIAL_IDENTIFY_OUT rather
than to stdout, because Rosmaster_Lib prints a connection banner to stdout the
moment a port opens and would otherwise be captured as part of the answer.

Exit status is 0 whether or not a port is identified; a missing or empty output
file means "none answered" and the caller decides what to do about it.
"""

import glob
import json
import os
import sys
import time

CANDIDATE_GLOBS = ("/dev/ttyUSB*", "/dev/myserial")
BY_ID = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
OUT_PATH = os.environ.get("SERIAL_IDENTIFY_OUT", "/tmp/rosmaster_serial_port")
OPEN_SETTLE_S = float(os.environ.get("SERIAL_IDENTIFY_SETTLE_S", "0.6"))
ATTEMPTS = int(os.environ.get("SERIAL_IDENTIFY_ATTEMPTS", "3"))


def log(**fields):
    sys.stderr.write("SERIAL_IDENTIFY " + json.dumps(fields, sort_keys=True) + "\n")
    sys.stderr.flush()


def candidates():
    """Candidate ports, most specific first, de-duplicated by real path."""
    found = []
    if os.path.exists(BY_ID):
        found.append(BY_ID)
    for pattern in CANDIDATE_GLOBS:
        found.extend(sorted(glob.glob(pattern)))

    seen = set()
    unique = []
    for path in found:
        try:
            key = os.path.realpath(path)
        except OSError:
            key = path
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def probe(port):
    """Return a truthy identity dict if this port answers as a Rosmaster board."""
    from Rosmaster_Lib import Rosmaster

    bot = None
    try:
        bot = Rosmaster(com=port, debug=False)
        bot.create_receive_threading()
        bot.set_auto_report_state(True, forever=False)
        # The board reports asynchronously, so give it a moment to speak before
        # concluding it is silent.
        time.sleep(OPEN_SETTLE_S)

        version = bot.get_version()
        voltage = bot.get_battery_voltage()
        # get_version returns 0 or a falsy value when nothing answered. A real
        # board reports a nonzero version, and usually a plausible pack voltage.
        plausible = bool(version) and float(version) > 0
        return {
            "version": version,
            "voltage": voltage,
            "answered": plausible,
        }
    finally:
        if bot is not None:
            try:
                bot.set_car_motion(0, 0, 0)
            except Exception:
                pass
            try:
                del bot
            except Exception:
                pass


def main():
    ports = candidates()
    log(event="scan", ports=ports)
    if not ports:
        log(event="no_candidates")
        return 0

    for port in ports:
        # Probing means opening, and opening means holding. The LiDAR lives on
        # one of these adapters and its driver cannot bind a port this process
        # has open, so stop at the first port that answers and never linger on
        # one that does not belong to us.
        for attempt in range(1, ATTEMPTS + 1):
            try:
                result = probe(port)
            except Exception as exc:  # noqa: BLE001 - any failure means "not this port"
                log(event="probe_error", port=port, attempt=attempt,
                    error=type(exc).__name__, detail=str(exc)[:200])
                continue

            log(event="probe", port=port, attempt=attempt, **result)
            if result["answered"]:
                log(event="identified", port=port, version=result["version"],
                    voltage=result["voltage"], out=OUT_PATH)
                with open(OUT_PATH, "w", encoding="utf-8") as handle:
                    handle.write(port)
                return 0

    log(event="unidentified", tried=ports,
        hint="no port reported a firmware version; check power and wiring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
