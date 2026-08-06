#!/usr/bin/env bash
# No -e: the supervisor loop at the bottom must survive its child dying,
# which is the whole point of having it.
set -o pipefail

if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
  echo "Restoring Python stdlib archive"
  rm -rf /usr/lib/python3.10
  tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
fi

source /opt/ros/humble/setup.bash

echo "rosmaster-a1-web-remote starting"
echo "PORT=${PORT:-8091}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
echo "web_remote.py sanity:"
wc -l /app/web_remote.py
tail -n 8 /app/web_remote.py
# Generate a self signed certificate so the remote can also be served over
# TLS. Browsers only expose the Gamepad API to secure contexts, so over plain
# HTTP to the car's address the Xbox controller is invisible to the page no
# matter how well it is connected. HTTPS makes the origin secure; the operator
# accepts the warning once per machine.
TLS_CERT="${TLS_CERT:-/app/webremote-cert.pem}"
TLS_KEY="${TLS_KEY:-/app/webremote-key.pem}"
export TLS_CERT TLS_KEY
if [[ ! -f "${TLS_CERT}" || ! -f "${TLS_KEY}" ]]; then
  # Every address the car is reachable on goes in the SAN, since browsers
  # ignore the legacy common name entirely.
  car_ips=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | sed 's/^/IP:/' | paste -sd, -)
  san="DNS:localhost,IP:127.0.0.1${car_ips:+,${car_ips}}"
  echo "Generating self signed TLS certificate with SAN ${san}"
  if openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -subj "/CN=rosmaster-a1-remote" -addext "subjectAltName=${san}" \
      -keyout "${TLS_KEY}" -out "${TLS_CERT}" >/dev/null 2>&1; then
    echo "TLS certificate ready at ${TLS_CERT}"
  else
    echo "TLS certificate generation failed; continuing with plain HTTP only" >&2
  fi
fi

# Supervised like the base and lidar services' processes: the runtime's
# recurring stdlib deletion (or any other fatal error) costs a ~5 s in-place
# relaunch here instead of a ~40 s container restart, and the exit status and
# traceback land in this container's log where the NEXT death is diagnosable.
# The flock serializes restores with any concurrent one; see the base
# service's entrypoint for the full story.
restore_stdlib() {
  if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
    (
      flock 9
      rm -rf /usr/lib/python3.10
      tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
    ) 9>/tmp/python-stdlib-restore.lock
  fi
}

attempt=0
backoff=5
while true; do
  attempt=$((attempt + 1))
  restore_stdlib
  python3 -X faulthandler -u /app/web_remote.py
  echo "WEB_REMOTE_SUPERVISOR exited status=$? attempt=${attempt}; restarting in ${backoff}s" >&2
  sleep "${backoff}"
  backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
done
