FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PORT=8091

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpython3.10-stdlib \
    python3 \
    python3-minimal \
    python3.10-minimal \
    python3-numpy \
    python3-pil \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

# Package the Python stdlib the same way the base and lidar services do: the
# Wendy runtime recurrently deletes /usr/lib/python3.10 on the device, and
# this service was the only one of the three with neither a restore archive
# nor a supervisor -- so a lazy import landing after a deletion (a PIL
# plugin, an ssl or encodings submodule) killed the process and cost a ~40 s
# runtime container restart, felt in the field as the car going deaf
# mid-drive. The zip is the fallback pure-Python path; the tarball is what
# the entrypoint's restore_stdlib untars before every (re)launch.
RUN python3 - <<'PY'
import os
import zipfile

stdlib = "/usr/lib/python3.10"
zip_path = "/usr/lib/python310.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for root, dirs, files in os.walk(stdlib):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for filename in files:
            if filename.endswith((".py", ".txt")):
                path = os.path.join(root, filename)
                archive.write(path, os.path.relpath(path, stdlib))
PY
RUN tar -C /usr/lib -czf /opt/python3.10-stdlib.tar.gz python3.10

WORKDIR /app
COPY app/server.py /app/web_remote.py
# Generate the TLS certificate at build time, not at container start.
#
# Browsers only hand the Gamepad API to a secure context, so the remote has to
# be reachable over HTTPS for the Xbox controller to work at all. A self signed
# certificate means one trust prompt per machine, which is fine, but only if the
# certificate stays the same. Minting it in the entrypoint gave every container
# start a new fingerprint and so a fresh warning every restart, which is exactly
# the friction that makes an operator reach for a workaround instead.
#
# Baked here it is stable for the life of the image. SAN covers localhost and
# any .local mDNS name, so the image is not tied to one car; browsers ignore
# the legacy common name entirely.
RUN openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -subj "/CN=rosmaster-a1-remote" \
      -addext "subjectAltName=DNS:localhost,DNS:*.local,IP:127.0.0.1" \
      -keyout /app/webremote-key.pem -out /app/webremote-cert.pem \
    && chmod 600 /app/webremote-key.pem

COPY app/entrypoint.sh /app/entrypoint.sh
COPY app/static/ /app/static/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
