#!/usr/bin/env bash
# Install Compose V2, docker-group access, and NVIDIA Container Toolkit for make kernel.
set -euo pipefail

COMPOSE_PACKAGE="docker-compose-v2"
TOOLKIT_KEY_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
TOOLKIT_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
TOOLKIT_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
TOOLKIT_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"

if [[ "$(id -u)" -ne 0 ]]; then
  printf 'run with sudo: sudo %s\n' "$0" >&2
  exit 1
fi

target_user="${SUDO_USER:-}"
if [[ -z "${target_user}" || "${target_user}" == "root" ]]; then
  printf 'SUDO_USER must be a non-root login user\n' >&2
  exit 1
fi

if ! getent group docker >/dev/null; then
  printf 'docker group is missing; install docker.io first\n' >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg "${COMPOSE_PACKAGE}"
usermod -aG docker "${target_user}"

if [[ ! -f "${TOOLKIT_LIST}" ]]; then
  curl -fsSL "${TOOLKIT_KEY_URL}" | gpg --dearmor -o "${TOOLKIT_KEYRING}"
  curl -fsSL "${TOOLKIT_LIST_URL}" \
    | sed "s#deb https://#deb [signed-by=${TOOLKIT_KEYRING}] https://#g" \
    > "${TOOLKIT_LIST}"
  apt-get update
fi

apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

if command -v systemctl >/dev/null && systemctl is-system-running >/dev/null 2>&1; then
  systemctl restart docker
else
  service docker restart
fi

printf 'host docker ready: %s, docker group for %s, nvidia runtime\n' \
  "${COMPOSE_PACKAGE}" "${target_user}"
printf 'open a new shell or run: sg docker -c '\''make kernel'\''\n'
