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
# The model comes from this app's own `ollama` service, not from whatever happened to be
# on the device. Wait for that sibling and adopt whichever model it pulled, so the two
# services cannot disagree about the name.
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
wait_for_model_runtime() {
  i=0
  while [ "$i" -lt 900 ]; do
    tags=$(curl -fsS --max-time 5 "$OLLAMA_HOST/api/tags" 2>/dev/null)
    if [ -n "$tags" ]; then
      # Prefer Nemotron when present: NemoClaw may have pulled a second model into the
      # shared volume, and taking whatever the API lists first picked the wrong one.
      m=$(printf '%s' "$tags" | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); ms=[x["name"] for x in (d.get("models") or [])]
    pref=[n for n in ms if "nemotron" in n.lower()]
    print((pref or ms or [""])[0])
except Exception: print("")' 2>/dev/null)
      [ -n "$m" ] && { echo "$m"; return 0; }
    fi
    i=$((i + 5)); sleep 5
  done
  echo ""
}
say "waiting for this app's model runtime (the ollama service)"
DISCOVERED_MODEL="$(wait_for_model_runtime)"
if [ -n "$DISCOVERED_MODEL" ]; then
  export NEMOCLAW_MODEL="${NEMOCLAW_MODEL:-$DISCOVERED_MODEL}"
  echo "PASS  model runtime ready, serving $NEMOCLAW_MODEL"
else
  export NEMOCLAW_MODEL="${NEMOCLAW_MODEL:-nemotron-3-nano:4b}"
  echo "FAIL  this app's model runtime never came up; onboarding will abort"
fi
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

# OpenShell writes per-sandbox JWTs under $HOME/.local/state and then bind-mounts them
# into the sandbox container. $HOME is on the container's own overlay layer, and a nested
# daemon cannot bind-mount a file off that overlay into its own container:
#   error mounting ".../sandbox.jwt" to rootfs at "/etc/openshell/auth/sandbox.jwt"
#   flags=MS_BIND|MS_REC
# Back that one directory with the ext4 persist volume, keeping the path identical. Only
# state lands there, never executables, so the volume's noexec does not bite.
mkdir -p /workspace/openshell-state "$HOME/.local/state"
if mountpoint -q "$HOME/.local/state" 2>/dev/null; then
  echo "state dir already bind-mounted"
elif mount --bind /workspace/openshell-state "$HOME/.local/state" 2>/dev/null; then
  echo "PASS  bound \$HOME/.local/state onto the ext4 volume (nested bind-mounts need this)"
else
  echo "WARN  could not bind state dir; sandbox container may fail to start"
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

  # OpenShell's sandbox records live on the persist volume while the JWT it bind-mounts is
  # written under the container's own overlay. A redeploy therefore resurrects a sandbox id
  # whose token file no longer exists, and the container fails with
  # "error mounting .../sandbox.jwt ... not a directory". Wipe the state tree so onboarding
  # starts from nothing; --fresh alone does not reach the gateway-side records.
  if [ "${NEMOCLAW_RESET_STATE:-1}" = "1" ]; then
    nemoclaw "$SANDBOX" destroy --yes >/dev/null 2>&1 || true
    rm -rf "$HOME/.local/state/openshell" 2>/dev/null
    echo "reset OpenShell state (stale sandbox ids cannot survive into this run)"
  fi

  # An aborted attempt leaves half-written TLS material and an orphaned sandbox, and both
  # make every later attempt fail with a different error. Clear them before onboarding.
  rm -rf "$HOME/.local/state/nemoclaw/openshell-docker-gateway-$GW_PORT/tls" 2>/dev/null
  openshell sandbox delete "$SANDBOX" >/dev/null 2>&1 || true

  say "onboarding options available in this build"
  nemoclaw onboard --help 2>&1 | head -80
  say "gateway/forward related environment knobs"
  nemoclaw --help 2>&1 | head -30

  # NemoClaw sizes the model against *available* GPU memory, so our own model runtime
  # holding 24 GB resident makes it reject the model we already have and fall back to one
  # we do not: "Requested Ollama model ... is unlikely to fit currently available GPU
  # memory; falling back to 'qwen3.5:9b'". Unload first; Ollama reloads on the next call.
  say "unloading the model so onboarding sees free GPU memory"
  curl -s --max-time 60 "${OLLAMA_HOST}/api/generate" \
    -d "{\"model\":\"$NEMOCLAW_MODEL\",\"keep_alive\":0}" >/dev/null 2>&1 \
    && echo "requested unload of $NEMOCLAW_MODEL"
  sleep 10
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU memory free now: ${free_mb:-unknown} MB"

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
  NEMOCLAW_YES=1 \
    nemoclaw onboard --non-interactive --yes --no-gpu --fresh --name "$SANDBOX" \
      --control-ui-port "$UI_PORT" 2>&1 | tail -30

  say "openshell gateway log (the real error behind a failed forward)"
  find "$HOME/.local/state/nemoclaw" -name 'openshell-gateway.log' -exec tail -40 {} \; 2>/dev/null \
    || echo "no gateway log found"
  say "ports already listening in this network namespace"
  ss -ltnp 2>/dev/null | head -15 || true

  say "non-interactive agent invocation surface"
  nemoclaw "$SANDBOX" agent --help 2>&1 | head -25 || true

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

# ---------------------------------------------------------------- verdict
# No fallback. An earlier version of this script quietly set up plain OpenClaw whenever
# the OpenShell sandbox failed, so every run printed healthy-looking PASS lines while the
# thing the sample exists to demonstrate had not started at all. That masked the real
# failure for days. If the sandbox is not Ready, say so loudly and stop.
if nemoclaw list 2>/dev/null | grep -q "$SANDBOX"; then
  AGENT_CMD="nemoclaw launch $SANDBOX"
  echo "PASS  NemoClaw sandbox '$SANDBOX' exists"
else
  AGENT_CMD=""
  cat <<'BANNER'

  ############################################################
  #  NEMOCLAW DID NOT COME UP                                #
  #                                                          #
  #  The OpenShell sandbox never reached Ready, so no agent  #
  #  is running. Everything below this line is diagnostics,  #
  #  not a working system. Do not demo this.                 #
  ############################################################

BANNER
  nemoclaw "$SANDBOX" doctor 2>&1 | tail -25 || true
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

# ---------------------------------------------------------------- diagnostics
say "sandbox token state (why the nested bind-mount fails)"
echo "XDG_STATE_HOME=${XDG_STATE_HOME:-unset}  HOME=$HOME"
echo "state dir mount: $(findmnt -no SOURCE,FSTYPE,TARGET "$HOME/.local/state" 2>/dev/null || echo 'not a mountpoint')"
find "$HOME/.local/state/openshell" -maxdepth 4 2>/dev/null | head -20 || echo "no openshell state tree"
find "$HOME/.local/state/openshell" -name 'sandbox.jwt' -exec ls -l {} \; 2>/dev/null | head -5 \
  || echo "no sandbox.jwt anywhere under the state tree"
echo "gateway processes:"; ps -eo pid,user,args 2>/dev/null | grep -iE 'openshell|gateway' | grep -v grep | head -5

say "agent CLI headless surface"
openclaw --help 2>&1 | grep -iE 'print|headless|non-inter|exec|run |-p,|--prompt' | head -12

# ---------------------------------------------------------------- cart pole
# The headline task: ask the on-device model to solve Cart Pole and keep the artifacts.
cartpole_is_solved() {
  python3 -c "import json,sys;d=json.load(open('/workspace/cartpole/results.json'));sys.exit(0 if d.get('best_steps',0)>=200 else 1)" 2>/dev/null
}
if [ "${RUN_CARTPOLE:-0}" = "1" ] && [ -n "$AGENT_CMD" ] && ! cartpole_is_solved; then
  # Opt-in only, and only when a sandbox actually exists.
  say "asking the agent to solve Cart Pole"
  /usr/local/bin/cartpole 2>&1 | tail -40 || echo "cart pole run did not finish"
fi

# Serve the artifacts so they can be collected without a TTY.
say "serving artifacts on port 8088"
(cd /workspace && python3 -m http.server 8088 >/dev/null 2>&1 &) \
  && echo "artifacts: http://<device>:8088/cartpole/"

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
