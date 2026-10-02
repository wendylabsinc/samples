"""Non-root hardware check for Wendy's direct NAN app entitlement."""

import json
import os
import secrets
import signal
import socket
import time


socket_path = os.environ["WENDY_NAN_SOCKET"]
client_dir = os.environ["WENDY_NAN_CLIENT_DIR"]
ndi = os.environ["WENDY_NAN_NDI"]
local_path = f"{client_dir}/smoke-{secrets.token_hex(8)}"
running = True

# Wendy's OCI path currently starts this image as root despite Docker USER.
# Drop explicitly before the first socket or interface check so this probe
# actually proves the entitlement's supplementary GID works for a non-root app.
if os.getuid() == 0:
    os.setgroups([2001])
    os.setgid(1000)
    os.setuid(1000)
if os.getuid() != 1000 or 2001 not in os.getgroups():
    raise RuntimeError("NAN entitlement probe is not running as uid 1000 with gid 2001")


def stop(_signal, _frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def interface_ready():
    """An idle NDI has no carrier or link-local address until its first NDP."""
    try:
        with open(f"/sys/class/net/{ndi}/flags", encoding="ascii") as flags:
            return bool(int(flags.read().strip(), 16) & 1)
    except FileNotFoundError:
        return False


with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as control:
    control.settimeout(3)
    control.bind(local_path)
    try:
        control.connect(socket_path)
        control.send(b"PING")
        reply = control.recv(4096).decode("ascii", errors="replace").strip()
        if reply != "PONG":
            raise RuntimeError(f"unexpected supplicant reply: {reply!r}")
        for _ in range(40):
            if interface_ready():
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"{ndi} is missing or administratively down")
        print(json.dumps({"result": "PASS", "uid": os.getuid(), "groups": os.getgroups(),
                          "ndi": ndi, "socket": socket_path, "client_dir": client_dir}), flush=True)
        while running:
            time.sleep(0.25)
    finally:
        os.unlink(local_path)
