# Security notes

Read this before deploying to anything you care about. Two of this sample's
entitlements are deliberately powerful, and both are visible in `wendy.json` rather
than hidden in the image.

## `admin` is a full-control grant

The `admin` entitlement bind-mounts the Wendy agent's local control socket into the
container. That socket has no authentication of its own: the entitlement mount is the
entire trust boundary.

Anything running in this container can therefore start, stop and **delete** any app on
the device, read all telemetry, exec into any other container, and trigger operating
system and agent updates. An agent that is prompted adversarially, or that follows an
instruction embedded in a web page or a file it reads, can wipe the device.

Deploy this to trusted, first-party devices only.

To run without device control, drop the `admin` entitlement. The agent still works,
still runs Nemotron locally, and still uses the Jetson Agent Skills; it simply cannot
change the device it is on.

## `build` is privileged-equivalent

NemoClaw's onboarding requires OpenShell's Docker compute driver, so this app runs a
nested Docker daemon. The `build` entitlement grants the namespace, mount and cgroup
privileges that needs, which in practice is the full `--privileged` capability set. It
carries container-to-host escape surface and, like `admin`, belongs only on devices you
trust.

A narrower entitlement scoped to nested rootless runtimes is in development. When it
lands, this sample should move to it.

## Credentials are stored unencrypted

Any provider credentials, messaging channel tokens, or agent session tokens live in the
persisted `/root` volume, protected only by the container's root user and the device's
filesystem. There is no encryption at rest and no key sealing.

Treat physical or root access to the device as equivalent to exposure of those
credentials. Revoke tokens before decommissioning, reassigning or reflashing a device.
Note that the volume outlives the container: `wendy device apps remove` deletes the app
but not its persisted data unless you pass `--delete-volumes`.

## Network exposure

The app uses host networking so the agent can reach a local model server and the rest of
your fleet. OpenShell binds its gateway and policy proxy to the loopback interface by
default. Do not publish those ports to a network; tunnel instead:

```bash
ssh -N -L 18789:127.0.0.1:18789 <user>@<host>
```

## Reporting

Found something wrong here? Open an issue on this repository. For vulnerabilities in
NemoClaw, OpenShell or the Jetson Agent Skills themselves, report to NVIDIA.
