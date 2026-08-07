# NVIDIA NemoClaw as a WendyOS app.
#
# Ubuntu 24.04 is NemoClaw's primary tested platform, so the app image matches it
# rather than the host. WendyOS itself is a minimal Yocto system with a read-only root
# filesystem: the whole NemoClaw stack lives inside this container, and the entitlements
# in wendy.json are what it is allowed to touch.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# NemoClaw's documented prerequisites (Node 22.19+, npm 10+, python3, zstd, binutils
# for `strings`), plus iptables for the nested Docker daemon and asciinema/tmux so a
# session can be recorded on the device.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ca-certificates curl git jq python3 python3-venv zstd binutils \
       iptables iproute2 procps util-linux sudo asciinema tmux \
  && rm -rf /var/lib/apt/lists/*

# Node 22 from NodeSource: the Ubuntu archive is older than NemoClaw requires.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && rm -rf /var/lib/apt/lists/* \
  && node --version && npm --version

# Docker Engine. NemoClaw's onboarding uses OpenShell's Docker compute driver and
# explicitly rejects Podman ("Podman is not supported for this NemoClaw integration
# path"), including the podman-docker shim, so a real dockerd is required. It runs
# nested inside this container; the `build` entitlement grants the namespace, mount and
# cgroup privileges that needs.
RUN curl -fsSL https://get.docker.com | sh \
  && rm -rf /var/lib/apt/lists/*

# Mount destinations must exist in the image. WendyOS does not create a missing
# bind-mount target, and the container fails at task creation if one is absent.
RUN mkdir -p /workspace /var/lib/docker /root

COPY init.sh /usr/local/bin/nemoclaw-init
RUN chmod +x /usr/local/bin/nemoclaw-init

# Runs as root deliberately: dockerd needs it, and the admin entitlement bind-mounts a
# root-owned control socket.
WORKDIR /workspace
CMD ["/usr/local/bin/nemoclaw-init"]
