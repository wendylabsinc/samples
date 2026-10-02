# time-browser

Browses `_wendytime._tcp` using DNS-SD, and every 10 seconds connects
to each resolved instance, logging its answer and host. Prints `discovered`
on first sight and `lost` when records expire.

Records arrive identically from mesh peers (agent catalog projection) and
physical-LAN devices (agent LAN projection) — no app-side distinction.

- `main.go` — query loop + tracker with TTL expiry + 10s HTTP poll loop
  (tracks only `_wendytime._tcp` records — the agent projects lab-LAN
  records alongside mesh ones, and polling printers/workstations is noise)
- `browse.go` — resolves every instance in a response, including multiple
  services sharing a host address and records split across packets. Missing
  SRV, TXT and A records are queried by their actual DNS type.
- `wendy.json` — mesh network mode, no published ports (outbound only)

Set `TIME_BROWSER_SERVICE_TYPE=_http._tcp` to observe the web sample. This mode
accepts only `WendyTimeWeb-` instances and logs the HTML response. Snapshots
are copied before HTTP polling so mDNS refreshes cannot race the poller.
See [the mesh tour](../MESH-TOUR.md) for direct deployment and cleanup.

```bash
docker build -t wendy-time-browser .
docker network create wendydemo
docker run -d --name pub --network wendydemo wendy-time-publisher
docker run --rm --network wendydemo wendy-time-browser   # Ctrl-C to stop
```

Note: the publisher's hashicorp/mdns sends no goodbye on shutdown, so loss surfaces via
TTL expiry (~130s here). On Wendy hardware, container stop also withdraws
the catalog record agent-side.
