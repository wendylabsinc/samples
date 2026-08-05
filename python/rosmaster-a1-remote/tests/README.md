# Tests

Two independent suites live here. Neither depends on the other.

## Python: web remote HTTP API (`tests/python`)

Exercises `rosmaster-a1-web-remote-wendy/app/server.py` off the robot, with
fake ROS packages (`tests/stubs`) standing in for `rclpy` and the ROS
message packages so the module can be imported without ROS 2 installed.
numpy and Pillow are real dependencies of real code under test (depth
colorization, JPEG encoding), so they must be actually installed, not
stubbed.

Setup, once:

```
python3 -m venv .venv
.venv/bin/pip install numpy Pillow
```

Run the suite from the repository root, with the venv active:

```
.venv/bin/python -m unittest discover -s tests/python -t .
```

Stdlib `unittest` only, no pytest, no fixtures library.

## JavaScript: web remote front end (`tests/web`)

Node's built-in test runner, stdlib only, in two layers.

`gamepad.test.mjs` covers
`rosmaster-a1-web-remote-wendy/app/static/gamepad.js`, the pure decision
layer: a Gamepad API reducer plus the stop path, disconnect and readout
policies. No DOM, no fetch, no page globals, so these are ordinary unit
tests.

`wiring.test.mjs` covers
`rosmaster-a1-web-remote-wendy/app/static/app.js`, the page script that
carries those decisions out. `harness.mjs` evaluates both static scripts in
a `node:vm` context with a fake DOM, fake gamepads and a fake car, the same
way the two script tags in `index.html` load them, so a test can call the
page's own functions and assert on the requests it made and the values it
rendered.

That second layer exists because the page script used to be inline in
`index.html`, and the only tests that could reach it read the file as text
and matched substrings. A reviewer's mutation run broke twelve real safety
behaviors with the whole suite still green. Do not add source scraping
assertions back; add a test that runs the code.

```
node --test tests/web/*.test.mjs
```
