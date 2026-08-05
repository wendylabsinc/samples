FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# The stock ROS 2 package pulls in librealsense2, which is what exposes the
# depth and both infrared streams. The HP60C could never do this: it computes
# depth on the module and hides the stereo pair, while the D435i publishes
# depth, infra1, infra2 and colour as separate topics.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-realsense2-camera \
      ros-humble-rmw-cyclonedds-cpp \
      v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
