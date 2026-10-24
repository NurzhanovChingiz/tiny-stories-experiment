#!/usr/bin/env bash
# Pull the pinned NVIDIA kernel image. No host GPU packages are vendored here.
# The Windows NVIDIA driver plus nvidia-container-toolkit must already exist.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_EXAMPLE="${ROOT}/docker/kernel/.env.example"
ENV_FILE="${ROOT}/docker/kernel/.env"
BASE_IMAGE="pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel"
UV_IMAGE="ghcr.io/astral-sh/uv:0.10.6"

if [[ ! -f "${ENV_FILE}" ]]; then
  uid="$(id -u)"
  gid="$(id -g)"
  awk -v uid="${uid}" -v gid="${gid}" '
    $0 ~ /^UID=/ { print "UID=" uid; next }
    $0 ~ /^GID=/ { print "GID=" gid; next }
    { print }
  ' "${ENV_EXAMPLE}" > "${ENV_FILE}"
  printf 'wrote %s\n' "${ENV_FILE}"
fi

docker pull "${UV_IMAGE}"
docker pull "${BASE_IMAGE}"
