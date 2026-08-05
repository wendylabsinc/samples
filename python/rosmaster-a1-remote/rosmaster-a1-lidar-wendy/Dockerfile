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

RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libpython3.10-stdlib \
    python3-colcon-common-extensions \
    python3-pip \
    python3-serial \
    python3.10-minimal \
    ros-humble-joy \
    ros-humble-ros2cli \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-default-generators \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    udev \
    v4l-utils \
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

WORKDIR /ros_ws
COPY ros_ws/src /ros_ws/src
RUN git clone --depth 1 https://github.com/YDLIDAR/YDLidar-SDK.git /tmp/YDLidar-SDK \
    && cmake -S /tmp/YDLidar-SDK -B /tmp/YDLidar-SDK/build \
    && cmake --build /tmp/YDLidar-SDK/build -j2 \
    && cmake --install /tmp/YDLidar-SDK/build \
    && rm -rf /tmp/YDLidar-SDK
RUN git clone --depth 1 --branch humble https://github.com/YDLIDAR/ydlidar_ros2_driver.git /ros_ws/src/ydlidar_ros2_driver
RUN . /opt/ros/humble/setup.sh \
    && colcon build --packages-select \
      ydlidar_ros2_driver \
      yahboomcar_msgs \
      yahboomcar_bringup \
      yahboomcar_ctrl

RUN mv /opt/ros/humble/bin/ros2 /opt/ros/humble/bin/ros2.real \
    && printf '%s\n' \
      '#!/usr/bin/env bash' \
      'if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then' \
      '  rm -rf /usr/lib/python3.10' \
      '  tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib' \
      'fi' \
      'exec /opt/ros/humble/bin/ros2.real "$@"' \
      > /opt/ros/humble/bin/ros2 \
    && chmod +x /opt/ros/humble/bin/ros2

WORKDIR /app
COPY app/sensor_probe.py /app/sensor_probe.py
COPY app/usb_reset.py /app/usb_reset.py
COPY app/port_census.py /app/port_census.py
COPY app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
