# PipeWire Audio

Plays a tone through PipeWire and reports which sinks the container can
see. Use it to check that audio really works on a device — including to a
paired Bluetooth speaker.

```sh
wendy run
```

## Why not `aplay`?

`aplay` and `speaker-test` open `/dev/snd` directly. They produce sound
from the built-in output whether or not PipeWire is reachable, so they
pass on a device whose audio stack is completely broken — and they can
never drive a Bluetooth speaker, which exists only as a node in the
PipeWire graph. This sample uses `paplay` against PipeWire's
PulseAudio-compatible socket, so a successful run means the graph is
genuinely wired up.

## Playing to a Bluetooth speaker

Pair the speaker first — no shell on the device required:

```sh
wendy device bluetooth list
wendy device bluetooth connect <address>
```

Then `wendy run`. The speaker appears in the sink list with a `bluez`
prefix:

```
=== sinks visible to this container ===
  52  bluez_output.78_2B_64_76_F3_CE.1  PipeWire  s16le 2ch 48000Hz  RUNNING
    ^ Bluetooth sink
```

## Reading the output

| What you see | What it means |
| --- | --- |
| `PULSE_SERVER=(unset)` | The host exposed no PulseAudio socket; nothing can play. |
| `(none -- the graph is empty)` | The container reached a PipeWire instance with no session manager, so it has no devices. |
| Sinks listed, tone audible | Working. |

The app needs only the `audio` entitlement. Pairing is handled by the
Wendy CLI, so no `bluetooth` entitlement is required just to play sound.

The app plays once and then idles. It does not exit: the runtime restarts
an exited app, which would replay the tone on a loop.
