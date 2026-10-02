#!/bin/sh
# Container entrypoint: private D-Bus + per-app avahi-daemon, then the app.
# Everything stays inside this container's network namespace (fully
# compartmentalized: no host D-Bus, no shared daemon). The agent observes
# avahi's multicast on the app bridge exactly like any other speaker.
set -eu
mkdir -p /run/dbus
rm -f /run/dbus/pid /run/avahi-daemon/pid 2>/dev/null || true
dbus-daemon --system --fork
# Allow co-binding :5353 with other stacks (agent bridge, libraries).
mkdir -p /etc/avahi
printf '[server]\ndisallow-other-stacks=no\n' > /etc/avahi/avahi-daemon.conf
# Mirror the agent's prerequisite ordering. The application itself makes a
# single registration attempt and contains no network startup retry.
sleep "${WENDY_TEST_AVAHI_DELAY:-0}"
avahi-daemon -D
attempt=0
until dbus-send --system --print-reply --dest=org.freedesktop.Avahi / \
  org.freedesktop.Avahi.Server.GetState 2>/dev/null | grep -q 'int32 2'; do
  attempt=$((attempt + 1))
  test "$attempt" -lt 30 || { echo 'Avahi failed to become ready' >&2; exit 1; }
  sleep 1
done
exec /usr/local/bin/time-web
