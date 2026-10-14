# ROCm kernel for notebooks/00_/gpu_pytorch.ipynb.
# Python 3.12 and HIP torch 2.9.1 already live in the base image at /opt/venv.
# Sync only the notebook group and keep that torch; do not install a PyPI torch wheel.
FROM ghcr.io/astral-sh/uv:0.10.6 AS uv

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --inexact --only-group notebook \
    && python -c "import torch; assert torch.version.hip, 'expected ROCm/HIP torch from /opt/venv'"

ARG UID=1000
ARG GID=1000

RUN mkdir -p \
        /app/cache \
        /home/app/.config/miopen \
        /tmp \
    && if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" app; fi \
    && if ! getent passwd "${UID}" >/dev/null; then \
         useradd --uid "${UID}" --gid "${GID}" --create-home --home-dir /home/app --shell /bin/bash app; \
       fi \
    && chown -R "${UID}:${GID}" \
        /app/cache \
        /home/app \
        /tmp

ENV HOME=/home/app \
    XDG_CACHE_HOME=/app/cache \
    UV_CACHE_DIR=/app/cache/uv \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2

USER ${UID}:${GID}

WORKDIR /app/notebooks/00_

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/notebooks/00_", "--ServerApp.token=", "--ServerApp.password="]
