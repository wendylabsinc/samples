# Drone slalom (Swift · MuJoCo Sim tab)

A Skydio X2 quadrotor flying a 5-gate slalom under a geometric controller, streamed live to
the Wendy Sandbox 🕹 Sim tab via `WendyMuJoCo`. Swift port of the MuJoCo reference
`../drone/starters/drone-slalom/mujoco_drone_race.py`.

## Build & run
Requires the sibling `swift-mujoco` package and MuJoCo installed (see that repo's README):

    export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig
    swift run DroneRace                       # streams to the Sim tab until stopped
    DRONE_MAX_STEPS=1500 swift run DroneRace  # bounded headless run (prints a summary)

Edit `defaultGates` / controller gains in `Sources/SlalomCore/` and rebuild.

> Dependency note: this sample uses a local path dependency on `../../../swift-mujoco`.
> Once `swift-mujoco` is published, switch to a versioned git-URL dependency.
