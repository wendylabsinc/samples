#!/bin/bash
# PID 1 for the NemoClaw app: supervise a nested Docker daemon, run NemoClaw's
# installer once, register the Wendy MCP server so the agent can operate the device,
# then stay alive so `wendy device attach nemoclaw` can open a session.
set -u

STATE=/root/.nemoclaw-app
LOG=/workspace/logs/install.log
SANDBOX="${NEMOCLAW_SANDBOX_NAME:-jetson}"
mkdir -p /workspace/logs /workspace/casts "$STATE"

say() { printf '\n===== %s =====\n' "$1"; }

# ---------------------------------------------------------------- dockerd
# Supervised rather than started once: if the daemon dies, every later agent action
# fails silently, which is worse than a restart loop we can see in the logs.
DOCKERD_PID=""
#
# The network flags are not optional. WendyOS mounts /proc/sys read-only even under the
# `build` entitlement, which grants every capability but is not the same as a privileged
# container. dockerd's default bridge setup writes /proc/sys/net/ipv4/ip_forward and dies:
#
#   failed to set IP forwarding '/proc/sys/net/ipv4/ip_forward' = '1': read-only file system
#
# --ip-forward=false stops it writing that sysctl, and --bridge=none skips the default
# bridge entirely. The app already runs with host networking, so sandbox containers reach
# the network directly rather than through a docker0 bridge.
start_dockerd() {
  dockerd --host=unix:///var/run/docker.sock \
          --data-root=/var/lib/docker \
          --storage-driver=overlay2 \
          --ip-forward=false \
          --iptables=false \
          --ip6tables=false \
          --bridge=none \
          >&2 2>&1 &
  DOCKERD_PID=$!
}
trap 'kill "$DOCKERD_PID" 2>/dev/null; exit 0' TERM INT

say "starting the nested Docker daemon"
start_dockerd
i=0
while [ "$i" -lt 60 ]; do
  docker info >/dev/null 2>&1 && break
  i=$((i + 1)); sleep 1
done
if docker info >/dev/null 2>&1; then
  echo "docker ready: $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
else
  echo "warning: docker did not become ready; NemoClaw onboarding will fail" >&2
fi

# ---------------------------------------------------------------- NemoClaw
# Restore a previous install from the volume, if we have one. Executables must land on
# the exec-capable image layer, so this is a copy rather than a mount or a symlink.
if [ -f /workspace/state/nchome.tar.gz ] && [ ! -d /opt/nchome/.nemoclaw ]; then
  mkdir -p /opt/nchome
  tar -xzf /workspace/state/nchome.tar.gz -C /opt/nchome \
    && echo "restored NemoClaw state from the persist volume" \
    && touch "$STATE/installed"
fi

say "filesystem exec flags (the tsc: Permission denied theory)"
for m in / /root /workspace /opt /tmp; do
  printf '%-12s %s\n' "$m" "$(findmnt -no FSTYPE,OPTIONS --target "$m" 2>/dev/null | head -1)"
done

if [ ! -f "$STATE/installed" ]; then
  say "installing NemoClaw (first start only; expect 5 to 15 minutes)"
  # The installer builds the CLI from a temporary clone and then executes what it built,
  # so TMPDIR must land on a filesystem mounted exec. On WendyOS both /tmp and the
  # persist volumes are noexec, and npm run build:cli dies with
  # "sh: 1: tsc: Permission denied". /opt lives on the container's own writable layer.
  # WendyOS mounts persist volumes noexec (confirmed: /root and /workspace are
  # ext4 rw,nosuid,noexec). NemoClaw installs into $HOME/.nemoclaw and then EXECUTES what
  # it installed, so its home cannot live on a volume: npm run build:cli dies with
  # "sh: 1: tsc: Permission denied". Symlinking the state directory onto a volume fails
  # the same way, and NemoClaw rejects symlinked state paths outright.
  #
  # So: install onto the container's own writable layer, then snapshot the state to the
  # volume so credentials and sandbox config survive a redeploy. Restore happens above,
  # before the install check.
  mkdir -p /opt/nctmp /opt/nchome
  (
    export TMPDIR=/opt/nctmp
    export NEMOCLAW_NON_INTERACTIVE=1 \
           NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
           NEMOCLAW_NO_EXPRESS=1 \
           NEMOCLAW_PROVIDER="${NEMOCLAW_PROVIDER:-ollama}" \
           NEMOCLAW_POLICY_MODE="${NEMOCLAW_POLICY_MODE:-suggested}" \
           NEMOCLAW_SANDBOX_NAME="$SANDBOX" \
           OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}" \
           HOME=/opt/nchome
    curl -fsSL https://www.nvidia.com/nemoclaw.sh \
      | bash -s -- --non-interactive --yes-i-accept-third-party-software
  ) >>"$LOG" 2>&1 && touch "$STATE/installed"

  # Snapshot the fresh install so the next redeploy does not pay for it again.
  if [ -f "$STATE/installed" ]; then
    mkdir -p /workspace/state
    tar -czf /workspace/state/nchome.tar.gz -C /opt/nchome . \
      && echo "saved NemoClaw state to the persist volume"
  fi
  tail -20 "$LOG"
fi

# ---------------------------------------------------------------- fleet tools
# The Wendy MCP server is what turns the agent from a chatbot into an operator: device
# inventory, app lifecycle, logs, telemetry. `mcp set` is idempotent, unlike `mcp add`,
# so this re-applies safely on every start.
if [ -n "${WENDY_AGENT_SOCKET:-}" ] && command -v nemoclaw >/dev/null 2>&1; then
  say "registering the Wendy MCP server with the agent"
  nemoclaw "$SANDBOX" exec -- \
    openclaw mcp set wendy '{"command":"wendy","args":["mcp","serve"]}' 2>&1 \
    || echo "warning: could not register the wendy MCP server (is onboarding finished?)" >&2
elif ! command -v nemoclaw >/dev/null 2>&1; then
  echo "note: nemoclaw is not installed yet, so there is nothing to register" >&2
else
  echo "note: no admin entitlement, so the agent has no device tools" >&2
fi

say "ready"
cat <<EOF
Attach a session:
  wendy device attach nemoclaw --device <your-device>.local

Then talk to the agent:
  nemoclaw launch $SANDBOX

Record it:
  asciinema rec /workspace/casts/demo.cast --idle-time-limit 2 --cols 120 --rows 34
EOF

# Supervise dockerd forever; PID 1 must stay alive for `wendy device attach`.
set +e
while true; do
  wait "$DOCKERD_PID"
  echo "warning: dockerd exited (code $?); restarting in 1s" >&2
  sleep 1
  start_dockerd
done
