FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188 AS python-stdlib

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends --reinstall \
      libpython3.10-stdlib \
      python3.10-minimal \
    && rm -rf /var/lib/apt/lists/*

FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# slam_toolbox pulls rviz plugins and Qt through its apt dependencies (~1.2 GB
# on top of ros-base); there is no slimmer package. The stdlib reinstall and
# the archive below are the same guard the base and lidar images carry: the
# runtime recurrently deletes /usr/lib/python3.10 and the keeper is Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpython3.10-stdlib \
    python3.10-minimal \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-slam-toolbox \
    && apt-get install -y --no-install-recommends --reinstall \
      libpython3.10-stdlib \
      python3.10-minimal \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-stdlib /usr/lib/python3.10/ /usr/lib/python3.10/

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

# The rosmaster-a1-maps persist volume mounts over /maps and hides this
# marker. A keeper that still sees it knows the volume is missing and says so
# (UNMOUNTED_MARKER in app/slam_keeper.py), instead of filling the container
# layer with sessions nobody will find again.
RUN mkdir -p /maps && touch /maps/.unmounted

WORKDIR /app
COPY app/slam_params.yaml /app/slam_params.yaml
COPY app/slam_args.sh /app/slam_args.sh
COPY app/slam_keeper.py /app/slam_keeper.py
COPY app/cyclone_env.sh /app/cyclone_env.sh
COPY app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
