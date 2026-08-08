#!/bin/bash
# PID 1 for the NemoClaw app: supervise a nested Docker daemon, run NemoClaw's
# installer once, register the Wendy MCP server so the agent can operate the device,
# then stay alive so `wendy device attach nemoclaw` can open a session.
set -u

# One HOME for everything in this container. NemoClaw's installer, its gateway and the
# agent must all agree, and it has to be an exec-capable filesystem: persist volumes are
# mounted noexec, so /root cannot hold an install that has to run.
export HOME=/opt/nchome
export PATH="/opt/nchome/.local/bin:$PATH"
mkdir -p "$HOME"

STATE=/opt/nchome/.nemoclaw-app
LOG=/workspace/logs/install.log
SANDBOX="${NEMOCLAW_SANDBOX_NAME:-jetson}"
mkdir -p /workspace/logs /workspace/casts "$STATE"

say() { printf '\n===== %s =====\n' "$1"; }

# ---------------------------------------------------------------- dockerd
# WendyOS mounts /proc/sys read-only, which stops dockerd writing ip_forward and stops it
# configuring container interfaces at all (it fails disabling IPv6 on the veth, and the
# container never starts). The `build` entitlement grants CAP_SYS_ADMIN and the container
# has its own mount namespace, so we can lift that ourselves rather than needing a
# platform change: remount /proc/sys read-write here.
PROC_SYS_RW=0
if mount -o remount,rw /proc/sys 2>/dev/null; then
  PROC_SYS_RW=1
  echo "PASS  remounted /proc/sys read-write"
else
  echo "WARN  could not remount /proc/sys read-write; falling back to restricted networking"
fi

# Supervised rather than started once: if the daemon dies, every later agent action
# fails silently, which is worse than a restart loop we can see in the logs.
DOCKERD_PID=""
start_dockerd() {
  if [ "$PROC_SYS_RW" = "1" ]; then
    # Normal daemon: bridge networking, iptables, forwarding. This is what OpenShell's
    # sandbox needs in order to get an address.
    dockerd --host=unix:///var/run/docker.sock \
            --data-root=/var/lib/docker \
            --storage-driver=overlay2 \
            >&2 2>&1 &
  else
    dockerd --host=unix:///var/run/docker.sock \
            --data-root=/var/lib/docker \
            --storage-driver=overlay2 \
            --ip-forward=false --iptables=false --ip6tables=false --bridge=none \
            >&2 2>&1 &
  fi
  DOCKERD_PID=$!
}
# Clean up the daemon's cgroup subtree on the way out. Without this, the cgroups the
# nested runtime created inside this app's delegated subtree survive the container and
# wedge the app id: every later deploy fails with
# "OCI runtime create failed: read status from sync socket", and removing the app does
# not clear it.
cleanup() {
  kill "$DOCKERD_PID" 2>/dev/null
  sleep 1
  for d in /sys/fs/cgroup/docker/*/ ; do [ -d "$d" ] && rmdir "$d" 2>/dev/null; done
  rmdir /sys/fs/cgroup/docker 2>/dev/null
  exit 0
}
trap cleanup TERM INT

say "starting the nested Docker daemon"
start_dockerd
i=0
while [ "$i" -lt 240 ]; do
  docker info >/dev/null 2>&1 && break
  i=$((i + 1)); sleep 1
done
if docker info >/dev/null 2>&1; then
  echo "docker ready: $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
else
  echo "warning: docker did not become ready; NemoClaw onboarding will fail" >&2
fi

# With host networking this app shares the device's port space, so OpenShell's defaults
# (gateway 8080, dashboard 18789) collide with whatever else the device runs. NemoClaw
# reads NEMOCLAW_GATEWAY_PORT and NEMOCLAW_DASHBOARD_PORT at module load, so every
# invocation needs them exported, not just `onboard`.
pick_port() {
  _p=$1; _end=$((_p + 200))
  while [ "$_p" -lt "$_end" ]; do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${_p}\$"; then
      echo "$_p"; return 0
    fi
    _p=$((_p + 1))
  done
  echo "$1"
}
export NEMOCLAW_GATEWAY_PORT="${NEMOCLAW_GATEWAY_PORT:-$(pick_port 18080)}"
export NEMOCLAW_DASHBOARD_PORT="${NEMOCLAW_DASHBOARD_PORT:-$(pick_port 18890)}"
export NEMOCLAW_OLLAMA_PROXY_PORT="${NEMOCLAW_OLLAMA_PROXY_PORT:-$(pick_port 11435)}"
# NemoClaw auto-picks the largest installed model, which frequently fails its completion
# probe on an edge device. Pin one that fits, overridable by the operator.
export NEMOCLAW_MODEL="${NEMOCLAW_MODEL:-qwen2.5:3b}"
GW_PORT="$NEMOCLAW_GATEWAY_PORT"; UI_PORT="$NEMOCLAW_DASHBOARD_PORT"
say "port selection (host networking shares the device's ports)"
echo "gateway=$GW_PORT dashboard=$UI_PORT"

# A previous run can leave a user-defined bridge claiming the same subnet as docker0.
# Both routes then exist, the stale one is linkdown, and every nested container loses all
# egress: it cannot even reach its own gateway. Prune before the daemon settles.
say "clearing stale docker networks (overlapping subnets black-hole nested traffic)"
docker network prune -f >/dev/null 2>&1 || true
for br in $(ip -o link show type bridge 2>/dev/null | awk -F': ' '{print $2}' | grep '^br-'); do
  docker network ls -q --filter "id=${br#br-}" | grep -q . || { ip link del "$br" 2>/dev/null && echo "removed stale bridge $br"; }
done

say "nested container networking smoke test"
if docker run --rm busybox sh -c 'nslookup registry.npmjs.org >/dev/null 2>&1' 2>/dev/null; then
  echo "PASS  nested container has working DNS and egress"
else
  echo "FAIL  nested container cannot resolve DNS; onboarding preflight will refuse"
fi

# ---------------------------------------------------------------- NemoClaw
# Restore a previous install from the volume, if we have one. Executables must land on
# the exec-capable image layer, so this is a copy rather than a mount or a symlink.
if [ -f /workspace/state/nchome.tar.gz ] && [ ! -d /opt/nchome/.nemoclaw ]; then
  mkdir -p /opt/nchome
  tar -xzf /workspace/state/nchome.tar.gz -C /opt/nchome 2>/dev/null
  # Trust the snapshot only if it actually contains a usable binary. An earlier version
  # set the installed marker from the tarball alone and then skipped a needed reinstall.
  if [ -x /opt/nchome/.local/bin/nemoclaw ]; then
    echo "restored NemoClaw state from the persist volume"
    touch "$STATE/installed"
  else
    echo "snapshot restored but nemoclaw binary missing; reinstalling"
    rm -f "$STATE/installed"
  fi
fi

say "filesystem exec flags (the tsc: Permission denied theory)"
for m in / /root /workspace /opt /tmp; do
  printf '%-12s %s\n' "$m" "$(findmnt -no FSTYPE,OPTIONS --target "$m" 2>/dev/null | head -1)"
done

install_attempt=0
while [ ! -f "$STATE/installed" ] && [ "$install_attempt" -lt 3 ]; do
  install_attempt=$((install_attempt + 1))
  # Do not start until the daemon is genuinely up: the installer aborts with "Docker is
  # installed but not reachable" and, before this loop existed, never retried for the
  # life of the container.
  if ! docker info >/dev/null 2>&1; then
    echo "waiting for docker before install attempt $install_attempt"
    j=0; while [ "$j" -lt 120 ]; do docker info >/dev/null 2>&1 && break; j=$((j + 1)); sleep 2; done
  fi
  say "installing NemoClaw (attempt $install_attempt; expect 5 to 15 minutes)"
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
           HOME=/opt/nchome  # inherited; kept explicit for the subshell
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
  [ -f "$STATE/installed" ] || { echo "install attempt $install_attempt failed; retrying"; sleep 5; }
done

# ---------------------------------------------------------------- onboarding
# The installer skips onboarding when its preflight objects. The only objection left on an
# AGX-class board is the missing NVIDIA CDI spec inside the nested Docker, which does not
# matter here: inference runs in a separate Wendy app that holds the `gpu` entitlement, so
# the sandbox itself needs no GPU passthrough. Onboard explicitly with --no-gpu.
export PATH="/opt/nchome/.local/bin:$PATH"
if command -v nemoclaw >/dev/null 2>&1 && ! nemoclaw list 2>/dev/null | grep -q "$SANDBOX"; then
  # The OpenShell installer can extract its gateway/sandbox binaries one directory too
  # deep (/usr/local/bin/openshell-sandbox/openshell-sandbox). NemoClaw then reports
  # "missing provider credential rewrite or MCP L7 policy support" and reinstalls in a
  # loop, because its feature gate requires the CLI, gateway and sandbox binaries under
  # one install root. Flatten them.
  for b in openshell-gateway openshell-sandbox; do
    if [ -d "/usr/local/bin/$b" ] && [ -x "/usr/local/bin/$b/$b" ]; then
      mv "/usr/local/bin/$b/$b" "/usr/local/bin/$b.tmp" \
        && rm -rf "/usr/local/bin/$b" \
        && mv "/usr/local/bin/$b.tmp" "/usr/local/bin/$b" \
        && echo "flattened misextracted $b"
    fi
  done

  # An aborted attempt leaves half-written TLS material and an orphaned sandbox, and both
  # make every later attempt fail with a different error. Clear them before onboarding.
  rm -rf "$HOME/.local/state/nemoclaw/openshell-docker-gateway-$GW_PORT/tls" 2>/dev/null
  openshell sandbox delete "$SANDBOX" >/dev/null 2>&1 || true

  say "onboarding options available in this build"
  nemoclaw onboard --help 2>&1 | head -80
  say "gateway/forward related environment knobs"
  nemoclaw --help 2>&1 | head -30

  say "onboarding sandbox '$SANDBOX' without GPU passthrough"
  NEMOCLAW_NON_INTERACTIVE=1 \
  NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
  NEMOCLAW_PROVIDER="${NEMOCLAW_PROVIDER:-ollama}" \
  NEMOCLAW_POLICY_MODE="${NEMOCLAW_POLICY_MODE:-suggested}" \
  NEMOCLAW_SANDBOX_NAME="$SANDBOX" \
  NEMOCLAW_MODEL="$NEMOCLAW_MODEL" \
  OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}" \
  CHAT_UI_URL="http://127.0.0.1:$UI_PORT" \
  NEMOCLAW_GATEWAY_PORT="$GW_PORT" \
    nemoclaw onboard --non-interactive --no-gpu --fresh --name "$SANDBOX" \
      --control-ui-port "$UI_PORT" 2>&1 | tail -30

  say "openshell gateway log (the real error behind a failed forward)"
  find "$HOME/.local/state/nemoclaw" -name 'openshell-gateway.log' -exec tail -40 {} \; 2>/dev/null \
    || echo "no gateway log found"
  say "ports already listening in this network namespace"
  ss -ltnp 2>/dev/null | head -15 || true

  say "sandbox status"
  nemoclaw list 2>&1 | head -10
fi

# ---------------------------------------------------------------- Jetson Agent Skills
if command -v nemoclaw >/dev/null 2>&1 && nemoclaw list 2>/dev/null | grep -q "$SANDBOX"; then
  if [ ! -f "$STATE/skills-installed" ]; then
    say "installing NVIDIA's Jetson Agent Skills into the sandbox"
    (cd /opt/jetson-device-skills && ./install.sh --targets nemoclaw \
        --nemoclaw-sandbox "$SANDBOX" 2>&1 | tail -20) && touch "$STATE/skills-installed"
  fi
fi

# ---------------------------------------------------------------- fallback agent
# NemoClaw's sandbox needs OpenShell's dashboard forward, which does not register in this
# environment, and its preflight also refuses boards under 8 GiB. Neither blocks the agent
# itself: OpenClaw, Nemotron and the Jetson skills are the parts that do the work, and
# WendyOS entitlements already provide the isolation OpenShell would. So if no sandbox
# exists, set up the direct path instead, and the app is usable either way.
if ! nemoclaw list 2>/dev/null | grep -q "$SANDBOX"; then
  say "OpenShell sandbox unavailable; setting up the direct agent path"

  mkdir -p "$HOME/.openclaw/skills"
  cp -r /opt/jetson-device-skills/skills/. "$HOME/.openclaw/skills/" 2>/dev/null || true
  cp -r /opt/jetson-bsp-skills/skills/.    "$HOME/.openclaw/skills/" 2>/dev/null || true
  echo "skills installed: $(ls "$HOME/.openclaw/skills" | wc -l)"
  ls "$HOME/.openclaw/skills" | head -10

  if [ -n "${WENDY_AGENT_SOCKET:-}" ]; then
    openclaw mcp set wendy '{"command":"wendy","args":["mcp","serve"]}' >/dev/null 2>&1 \
      && echo "PASS  wendy MCP server registered with OpenClaw" \
      || echo "WARN  could not register the wendy MCP server"
  fi

  if [ -z "${OLLAMA_HOST:-}" ]; then
    OLLAMA_HOST=http://127.0.0.1:11434
    if ! curl -fsS --max-time 3 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
      GW="$(ip route 2>/dev/null | awk '/^default/{print $3; exit}')"
      [ -n "$GW" ] && curl -fsS --max-time 3 "http://$GW:11434/api/tags" >/dev/null 2>&1 \
        && OLLAMA_HOST="http://$GW:11434"
    fi
    export OLLAMA_HOST
  fi
  # NemoClaw probes the model with a real completion. A model server can answer /api/tags
  # while /api/generate hangs (a known wedged-Ollama state), and onboarding then aborts
  # with "model unavailable". Warm the model so the probe succeeds.
  if [ -n "${NEMOCLAW_MODEL:-}" ]; then
    curl -s --max-time 120 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/generate" \
      -d "{\"model\":\"$NEMOCLAW_MODEL\",\"prompt\":\"hi\",\"stream\":false}" >/dev/null 2>&1 \
      && echo "PASS  model $NEMOCLAW_MODEL answered a completion probe" \
      || echo "WARN  model $NEMOCLAW_MODEL did not answer /api/generate; onboarding will abort"
  fi
  if curl -fsS --max-time 5 "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
    echo "PASS  model server reachable at ${OLLAMA_HOST:-http://127.0.0.1:11434}"
  else
    echo "WARN  no model server at ${OLLAMA_HOST:-http://127.0.0.1:11434}; deploy one before recording"
  fi
  AGENT_CMD="openclaw"
else
  AGENT_CMD="nemoclaw launch $SANDBOX"
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
  $AGENT_CMD

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
