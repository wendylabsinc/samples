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
start_dockerd() {
  dockerd --host=unix:///var/run/docker.sock \
          --data-root=/var/lib/docker \
          --storage-driver=overlay2 \
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
if [ ! -f "$STATE/installed" ]; then
  say "installing NemoClaw (first start only; expect 5 to 15 minutes)"
  (
    export NEMOCLAW_NON_INTERACTIVE=1 \
           NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
           NEMOCLAW_NO_EXPRESS=1 \
           NEMOCLAW_PROVIDER="${NEMOCLAW_PROVIDER:-ollama}" \
           NEMOCLAW_POLICY_MODE="${NEMOCLAW_POLICY_MODE:-suggested}" \
           NEMOCLAW_SANDBOX_NAME="$SANDBOX" \
           OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
    curl -fsSL https://www.nvidia.com/nemoclaw.sh \
      | bash -s -- --non-interactive --yes-i-accept-third-party-software
  ) >>"$LOG" 2>&1 && touch "$STATE/installed"
  tail -20 "$LOG"
fi

# ---------------------------------------------------------------- fleet tools
# The Wendy MCP server is what turns the agent from a chatbot into an operator: device
# inventory, app lifecycle, logs, telemetry. `mcp set` is idempotent, unlike `mcp add`,
# so this re-applies safely on every start.
if [ -n "${WENDY_AGENT_SOCKET:-}" ]; then
  say "registering the Wendy MCP server with the agent"
  nemoclaw "$SANDBOX" exec -- \
    openclaw mcp set wendy '{"command":"wendy","args":["mcp","serve"]}' 2>&1 \
    || echo "warning: could not register the wendy MCP server (is onboarding finished?)" >&2
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
