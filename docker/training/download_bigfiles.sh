#!/usr/bin/env bash
# Download the pinned ROCm training images and WSL host packages.
# Debs are stored once under data/raw/rocm-host and are not overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/data/raw/rocm-host"
ENV_EXAMPLE="${ROOT}/docker/training/.env.example"
ENV_FILE="${ROOT}/docker/training/.env"
BASE_IMAGE="rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1"
UV_IMAGE="ghcr.io/astral-sh/uv:0.10.6"

mkdir -p "${DEST}"

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

download() {
  local url="$1"
  local name="$2"
  local target="${DEST}/${name}"
  if [[ -f "${target}" ]]; then
    printf 'present %s\n' "${target}"
    return 0
  fi
  printf 'download %s\n' "${url}"
  rm -f "${target}.partial"
  curl --fail --location --retry 5 --retry-delay 2 --output "${target}.partial" "${url}"
  mv "${target}.partial" "${target}"
  printf 'saved %s\n' "${target}"
}

download \
  "https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb" \
  "amdgpu-install_7.2.1.70201-1_all.deb"
download \
  "https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb" \
  "rocdxg-roct_1.2.2_amd64.deb"
download \
  "https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb" \
  "rocdxg-amd-smi-lib_1.2.2_amd64.deb"

docker pull "${UV_IMAGE}"
docker pull "${BASE_IMAGE}"
