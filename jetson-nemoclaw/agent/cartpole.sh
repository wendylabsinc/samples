#!/bin/bash
# Ask the on-device model to solve Cart Pole, run what it writes, and keep the artifacts.
#
# The agent's brain is Nemotron running on this device. This script gives it the task,
# executes the program it produces, and feeds failures back for another attempt, which is
# what an agent loop does. Everything lands in /workspace/cartpole so it can be collected.
set -u

OUT=/workspace/cartpole
mkdir -p "$OUT"
OLLAMA="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${NEMOCLAW_MODEL:-nemotron-3-nano:30b}"
ATTEMPTS=${CARTPOLE_ATTEMPTS:-3}

TASK='Write a single self-contained Python 3 program that solves the classic Cart Pole
control problem WITHOUT gym, gymnasium, or any reinforcement learning library.

Use exactly this method, it is small and reliable:
  - State is [x, x_dot, theta, theta_dot]. Policy is linear: push right if
    dot(w, state) > 0 else push left, where w is a length-4 numpy vector.
  - Train by random search with hill climbing: start w = zeros. For up to 2000 trials,
    sample noise = numpy.random.randn(4) * 0.5, evaluate w + noise over one episode,
    and keep the candidate if it scores better than the best so far.
  - Stop training early as soon as an episode reaches 500 steps.

Physics, standard cart pole, implement it yourself as plain functions or a class where
every attribute you use is assigned in __init__:
  gravity 9.8, cart mass 1.0, pole mass 0.1, total mass 1.1, pole half-length 0.5,
  polemass_length 0.05, force 10.0, tau 0.02, Euler integration.
  Episode fails when abs(theta) > 12 degrees in radians, or abs(x) > 2.4.
  Episode caps at 500 steps.

Use EXACTLY these update equations. Getting the timestep scaling wrong is the usual
failure, every velocity update must be multiplied by tau:
  force = +10.0 if action else -10.0
  temp = (force + polemass_length * theta_dot**2 * sin_theta) / total_mass
  thetaacc = (gravity * sin_theta - cos_theta * temp) /
             (half_length * (4.0/3.0 - mass_pole * cos_theta**2 / total_mass))
  xacc = temp - polemass_length * thetaacc * cos_theta / total_mass
  then, in this order:
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
Reset each episode to small random values, numpy.random.uniform(-0.05, 0.05, 4).

Hard requirements:
1. HARD CAP: never exceed 2000 training trials. The whole program must finish in under
   4 minutes. No infinite or unbounded loops anywhere.
2. Print progress sparsely: only every 50th trial, as "trial <n> best <steps>".
   Never print more than 100 lines in total.
3. After training, run one greedy evaluation episode with the best w, recording x and
   theta at each step.
4. Render that evaluation episode to /workspace/cartpole/cartpole.mp4 with matplotlib
   (matplotlib.use("Agg"), FuncAnimation, FFMpegWriter, fps 50): draw the track, the cart
   as a rectangle, the pole as a line. If mp4 writing raises, fall back to
   /workspace/cartpole/cartpole.gif with PillowWriter. Wrap rendering in try/except and
   always produce one of the two files.
5. Write /workspace/cartpole/results.json with keys: trials_run, best_steps,
   final_eval_steps, solved (true when best_steps >= 500).
6. Only numpy, matplotlib, json and the standard library. No network. No argparse.
7. Define every attribute before use. Do not reference self.I or any moment of inertia.
8. Print "CARTPOLE DONE" as the very last line, after the files are written.

Output ONLY the Python code, no explanation, no markdown fences.'

extract_code() {
  python3 - "$1" <<'PY'
import re, sys
raw = open(sys.argv[1]).read()
# The model may wrap in fences or emit a reasoning preamble; take the largest code block
# if fenced, otherwise everything from the first import/def onward.
blocks = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.S)
code = max(blocks, key=len) if blocks else raw
m = re.search(r"^(?:import |from |#!|import\n)", code, re.M)
if m:
    code = code[m.start():]
sys.stdout.write(code)
PY
}

feedback=""
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "=== cart pole attempt $attempt of $ATTEMPTS (model: $MODEL) ==="
  prompt="$TASK"
  [ -n "$feedback" ] && prompt="$TASK

Your previous program failed. Fix it. The error was:
$feedback"

  python3 - "$OLLAMA" "$MODEL" "$OUT/raw-$attempt.txt" <<'PY' "$prompt"
import json, sys, urllib.request
ollama, model, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = sys.argv[4]
req = urllib.request.Request(
    f"{ollama}/api/generate",
    # think=false matters: these are hybrid reasoning models, and with a small budget the
    # entire allowance is spent in the thinking channel, leaving "response" empty.
    data=json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                     "options": {"temperature": 0.2, "num_predict": 16384}}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1800) as r:
    body = json.load(r)
text = body.get("response") or body.get("thinking") or ""
open(outfile, "w").write(text)
print(f"model returned {len(text)} characters "
      f"(response={len(body.get('response') or '')}, thinking={len(body.get('thinking') or '')})")
if not text:
    print("raw keys:", list(body.keys())[:12])
PY

  extract_code "$OUT/raw-$attempt.txt" > "$OUT/cartpole.py"
  echo "--- generated program, first lines:"; head -12 "$OUT/cartpole.py"

  if timeout 300 python3 "$OUT/cartpole.py" 2>&1 | head -c 200000 > "$OUT/run-$attempt.log"; [ "${PIPESTATUS[0]}" = "0" ]; then
    if grep -q "CARTPOLE DONE" "$OUT/run-$attempt.log"; then
      best=$(python3 -c "import json;print(json.load(open('$OUT/results.json')).get('best_steps',0))" 2>/dev/null || echo 0)
      if [ "${best:-0}" -ge 200 ]; then
        echo "PASS  balanced for $best steps on attempt $attempt"
        tail -5 "$OUT/run-$attempt.log"; ls -l "$OUT"; exit 0
      fi
      # Running to completion is not the same as working. A best of 1 step means the
      # physics integration is wrong, which is the mistake this model keeps making.
      feedback="The program ran but never balanced: best_steps=$best out of 500. Every
episode ends almost immediately, so the physics integration is wrong. Multiply every
velocity update by tau and use the exact equations given. Do not change the algorithm."
    else
      feedback="the program exited 0 but never printed CARTPOLE DONE"
    fi
  else
    feedback="$(tail -25 "$OUT/run-$attempt.log")"
  fi
  echo "attempt $attempt failed; feeding the error back to the model"
  echo "$feedback" | tail -5
done

echo "FAIL  cart pole did not complete in $ATTEMPTS attempts"
exit 1
