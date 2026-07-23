# Drone slalom (Swift · MuJoCo Sim tab)

A Skydio X2 quadrotor flying a 5-gate slalom under a geometric controller, streamed live to
the Wendy Sandbox 🕹 Sim tab via `WendyMuJoCo`. Swift port of the MuJoCo reference
`../drone/starters/drone-slalom/mujoco_drone_race.py`.

## Build & run
Depends on [`swift-mujoco`](https://github.com/wendylabsinc/swift-mujoco) (fetched
automatically) and MuJoCo installed locally (see that repo's README for the one-time
`install-mujoco.sh` step):

    export PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig
    swift run DroneRace                       # streams to the Sim tab until stopped
    DRONE_MAX_STEPS=1500 swift run DroneRace  # bounded headless run (prints a summary)

Edit `defaultGates` / controller gains in `Sources/SlalomCore/` and rebuild.

> Dependency note: this sample depends on `swift-mujoco` via git URL (branch `main`).
> Pin to a tagged release once one is cut for reproducible builds.
