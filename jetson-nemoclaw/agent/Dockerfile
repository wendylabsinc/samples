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
       openssh-client lsof \
       netcat-openbsd \
       python3-numpy python3-matplotlib ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# openssh-client is REQUIRED, not optional: OpenShell implements its dashboard port
# forward as an SSH tunnel (`ssh -o ProxyCommand="openshell ssh-proxy ..."`). Without the
# ssh binary the forward dies with "No such file or directory (os error 2)", onboarding
# times out waiting for the forward to register, and the sandbox is rolled back. NVIDIA's
# tested hosts all ship ssh, so their docs never mention it. lsof is used by the same
# forward code to check port ownership.

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

# OpenClaw itself, for the fallback path. NemoClaw is OpenClaw plus Nemotron plus the
# Jetson skills plus OpenShell; when OpenShell's sandbox cannot come up, the first three
# still give a complete on-device agent, sandboxed by WendyOS entitlements instead.
RUN npm install -g openclaw@latest && openclaw --version

# NVIDIA's Jetson Agent Skills, cloned at build time so the device needs no git access.
# install.sh is run after onboarding, once the sandbox exists to install them into.
RUN git clone --depth 1 https://github.com/NVIDIA-AI-IOT/jetson-device-skills.git \
      /opt/jetson-device-skills \
 && git clone --depth 1 https://github.com/NVIDIA-AI-IOT/jetson-bsp-skills.git \
      /opt/jetson-bsp-skills \
 && test -f /opt/jetson-device-skills/skills/jetson-memory-audit/SKILL.md

# Mount destinations must exist in the image. WendyOS does not create a missing
# bind-mount target, and the container fails at task creation if one is absent.
RUN mkdir -p /workspace /var/lib/docker /root

COPY cartpole.sh /usr/local/bin/cartpole
COPY init.sh /usr/local/bin/nemoclaw-init
RUN chmod +x /usr/local/bin/nemoclaw-init /usr/local/bin/cartpole

# Runs as root deliberately: dockerd needs it, and the admin entitlement bind-mounts a
# root-owned control socket.
WORKDIR /workspace
CMD ["/usr/local/bin/nemoclaw-init"]
