# time-publisher

Answers "what is the current time?" over HTTP (`GET /` →
`{"time":"...","host":"..."}`) and publishes a custom `_wendytime._tcp`
mDNS service, using the hashicorp/mdns library (raw multicast; co-binds
:5353 with the agent bridge on Linux).

- `main.go` — HTTP route (stdlib `net/http`) + publish + query responder
- `wendy.json` — mesh network mode + published TCP port 8080
- `Dockerfile` — multi-stage Go build (also used by `wendy run`)

On a Wendy device the agent observes this multicast on the app container
network and signs the record into the mesh catalog — no Avahi or D-Bus
involved. See `../MESH-TOUR.md`.

```bash
docker build -t wendy-time-publisher .
docker network create wendydemo
docker run -d --name pub --network wendydemo -p 127.0.0.1:8080:8080 wendy-time-publisher
curl http://127.0.0.1:8080/
```

For a run-specific service alias, set TIME_SERVICE_HOSTNAME. Without this
override, the app uses the agent-supplied WENDY_DEVICE_HOSTNAME.
