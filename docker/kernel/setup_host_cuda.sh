#!/usr/bin/env bash
# Create .venv-cuda with project extras plus CUDA torch from the official index.
# Torch stays out of the lockfile so the kernel image keeps its own wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${ROOT}/.venv-cuda"
TORCH_INDEX="https://download.pytorch.org/whl/cu132"
TORCH_PIN="torch==2.14.1"

cd "${ROOT}"
uv venv "${VENV}" --python 3.12
UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --inexact --group notebook --group inference --group profiling
UV_PROJECT_ENVIRONMENT="${VENV}" uv pip install --python "${VENV}/bin/python" \
  --index-url "${TORCH_INDEX}" \
  "${TORCH_PIN}"
"${VENV}/bin/python" -c "import torch; assert torch.version.cuda, 'expected CUDA torch in .venv-cuda'"
printf 'host CUDA venv ready at %s\n' "${VENV}"
