#!/bin/bash
# The sample's own model runtime. This exists so the sample owes nothing to the device it
# lands on: a clean Jetson has no model server, and an agent without one is furniture.
set -u

MODELS_DIR=${OLLAMA_MODELS:-/models}
mkdir -p "$MODELS_DIR"

# Pick a model that fits the board unless the operator pinned one. The Nemotron sizes
# match the table in the sample README: AGX class runs the 30B comfortably, an 8 GB Orin
# Nano does not and gets the 4B.
if [ -z "${NEMOCLAW_MODEL:-}" ]; then
  mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
  mem_gb=$((mem_kb / 1024 / 1024))
  if [ "$mem_gb" -ge 60 ]; then
    NEMOCLAW_MODEL="nemotron-3-nano:30b"
  else
    NEMOCLAW_MODEL="nemotron-3-nano:4b"
  fi
  echo "detected ${mem_gb} GB of memory; selected $NEMOCLAW_MODEL"
fi
export NEMOCLAW_MODEL

echo "starting ollama serve"
ollama serve &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null; exit 0' TERM INT

# Wait for the API rather than sleeping a guess.
i=0
until ollama list >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 120 ] && { echo "FAIL ollama did not become ready"; break; }
  sleep 1
done
ollama list >/dev/null 2>&1 && echo "PASS ollama API is up"

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$NEMOCLAW_MODEL"; then
  echo "PASS model $NEMOCLAW_MODEL already present"
else
  echo "pulling $NEMOCLAW_MODEL; this is the long part of a first run"
  if ollama pull "$NEMOCLAW_MODEL"; then
    echo "PASS pulled $NEMOCLAW_MODEL"
  else
    echo "FAIL could not pull $NEMOCLAW_MODEL"
  fi
fi

# Prove the model actually answers. A server can list a model and still hang on
# generation, and NemoClaw's onboarding probes with a real completion.
if ollama run "$NEMOCLAW_MODEL" "reply with the single word READY" >/tmp/probe 2>/dev/null; then
  echo "PASS completion probe: $(tr -d '\n' </tmp/probe | head -c 80)"
else
  echo "FAIL model did not answer a completion"
fi

echo "MODEL RUNTIME READY: $NEMOCLAW_MODEL on ${OLLAMA_HOST:-0.0.0.0:11434}"
wait "$SERVE_PID"
