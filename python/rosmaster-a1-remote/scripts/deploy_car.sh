#!/usr/bin/env bash
# Deploy the car's services with entitlements that match the hardware present.
#
# A Wendy serial entitlement naming a device that is not connected does not
# degrade, it hard-fails container creation:
#
#   serial device /dev/ttyUSB0 unavailable (need a real, connected tty node)
#
# The two CH340 adapters on this chassis renumber between boots and one of them
# has been dropping off the bus, so a wendy.json that is correct on Monday makes
# the app undeployable on Tuesday, over a cable rather than a code change. That
# has now broken a deploy three times, once taking motor control with it.
#
# So: ask the device which tty nodes actually exist, write only those into the
# serial entitlements, and deploy. The services already choose the right port
# at runtime by elimination, and the LiDAR service retries, so over-entitling
# is safe and under-entitling only costs a redeploy once the hardware returns.
#
# The four apps that used to deploy separately (each with its own wendy.json)
# are now one app, rosmaster-a1, with four services (base, lidar, realsense,
# web) sharing a single root wendy.json, so pruning runs once against that
# manifest instead of once per app directory. That also raises the stakes:
# with all four services in one app, an absent entitled tty now blocks that
# service's container for the whole app deploy, which makes pruning MORE
# important than it was with four separate apps.
#
# Division of labor for a no-args deploy (all four services):
#   - pruning above handles the absent-serial hard-fail when enumeration
#     succeeds, by dropping the offending entitlement before the single
#     group `wendy run` call, so no service ever tries to create a container
#     against a device that isn't there;
#   - when enumeration fails, pruning can't run, so the per-service fallback
#     below restores the old isolation: each service deploys on its own,
#     so an absent-serial hard-fail in one can't abort its siblings;
#   - --keep-going on the group call covers a different failure class
#     entirely (build/push failures), not on-device container-creation
#     hard-fails, in either of the above cases.
#
# Usage: scripts/deploy_car.sh <car-hostname>.local:50052 [service ...]

set -uo pipefail

DEVICE="${1:-${WENDY_DEVICE:-}}"
if [[ -z "${DEVICE}" ]]; then
  echo "usage: scripts/deploy_car.sh <car-hostname>.local:50052 [service ...]" >&2
  echo "   or: WENDY_DEVICE=<car-hostname>.local:50052 scripts/deploy_car.sh" >&2
  exit 2
fi
shift || true
SERVICES=("$@")

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

echo "Asking ${DEVICE} which serial nodes exist"
# Older CLIs exposed this as `wendy hardware capabilities` returning a
# {"capabilities": [...]} object; current CLIs use `wendy device hardware list`
# returning a bare array. Older agents (0.17.x) also omit serial entries
# entirely, which lands in the empty-result fallback below.
present=$(wendy --json device info --device "${DEVICE}" >/dev/null 2>&1 && \
          wendy --json device hardware list --device "${DEVICE}" 2>/dev/null | \
          python3 -c '
import json,sys
try:
    doc=json.load(sys.stdin)
except Exception:
    sys.exit(0)
caps=doc.get("capabilities",[]) if isinstance(doc,dict) else doc
for c in caps:
    p=c.get("device_path","")
    if p.startswith("/dev/ttyUSB"):
        print(p.rsplit("/",1)[-1])
' || true)

if [[ -z "${present}" ]]; then
  echo "Could not enumerate tty nodes from the device." >&2
  echo "Deploying with wendy.json as committed; if container creation fails with" >&2
  echo "'serial device ... unavailable', remove that device and retry." >&2
  echo "(A no-args deploy falls back to one 'wendy run' per service below, so" >&2
  echo "that hard-fail can't take its sibling services down with it.)" >&2
else
  echo "Present tty nodes: ${present//$'\n'/ }"
fi

if [[ -n "${present}" ]]; then
  PRESENT="${present}" python3 - "wendy.json" <<'PY'
import json, os, sys
path = sys.argv[1]
present = set(os.environ["PRESENT"].split())
doc = json.load(open(path))
changed = False
for name, svc in doc.get("services", {}).items():
    ents = svc.get("entitlements", [])
    kept = []
    for e in ents:
        if e.get("type") == "serial" and e.get("device") not in present:
            print("  dropping absent serial entitlement:", e.get("device"))
            changed = True
            continue
        kept.append(e)
    svc["entitlements"] = kept
if changed:
    json.dump(doc, open(path, "w"), indent=2)
    open(path, "a").write("\n")
PY
fi

if [[ ${#SERVICES[@]} -eq 0 && -z "${present}" ]]; then
  echo
  echo "Enumeration failed, so pruning could not drop absent-serial entitlements;" >&2
  echo "falling back to the per-service loop so a container-creation hard-fail in" >&2
  echo "one service can't abort the others." >&2
  SERVICES=(base lidar realsense web)
fi

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  echo
  echo "=== rosmaster-a1 (all services) ==="
  # --keep-going deploys the services whose builds/pushes succeed instead of
  # aborting the whole group (the moral equivalent of the old loop's
  # continue-past-failures) -- it only covers build/push failures. Absent-serial
  # hard-fails are what the pruning above is for; this group path only runs
  # once pruning has actually had a chance to drop them (enumeration
  # succeeded), so it's safe to fail as one unit here.
  wendy run --yes --detach --builder docker --keep-going --device "${DEVICE}" || \
    echo "  deploy FAILED" >&2
else
  for svc in "${SERVICES[@]}"; do
    echo
    echo "=== ${svc} ==="
    wendy run --yes --detach --builder docker --service "${svc}" --device "${DEVICE}" || \
      echo "  deploy of ${svc} FAILED" >&2
  done
fi

echo
echo "Note: entitlements may have been edited to match present hardware."
echo "Review with: git diff -- wendy.json"
