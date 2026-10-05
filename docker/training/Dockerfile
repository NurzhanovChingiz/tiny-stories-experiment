# ROCm/HIP training image. Python 3.12 and torch 2.9.1 come from the base.
# Do not install the project lock's torch or triton into this image.
FROM ghcr.io/astral-sh/uv:0.10.6 AS uv

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 AS deps

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

RUN --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --inexact --no-install-project --group notebook \
        --no-install-package torch --no-install-package triton \
    && python -c "import torch; assert torch.version.hip, 'expected ROCm/HIP torch from /opt/venv'"

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

COPY --from=uv /uv /uvx /bin/
COPY --from=deps /opt/venv /opt/venv

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
COPY src ./src
COPY docker/training/smoke.py ./docker/training/smoke.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --inexact --group notebook \
        --no-install-package torch --no-install-package triton \
    && python -c "import torch; assert torch.version.hip, 'expected ROCm/HIP torch from /opt/venv'"

ARG UID=1000
ARG GID=1000

RUN mkdir -p \
        /app/artifacts \
        /app/cache \
        /app/checkpoints \
        /data/datasets \
        /home/app/.config/miopen \
        /tmp \
    && if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" app; fi \
    && if ! getent passwd "${UID}" >/dev/null; then \
         useradd --uid "${UID}" --gid "${GID}" --create-home --home-dir /home/app --shell /bin/bash app; \
       fi \
    && chown -R "${UID}:${GID}" \
        /app/artifacts \
        /app/cache \
        /app/checkpoints \
        /data/datasets \
        /home/app \
        /tmp

ENV HOME=/home/app \
    XDG_CACHE_HOME=/app/cache \
    UV_CACHE_DIR=/app/cache/uv \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2

USER ${UID}:${GID}

CMD ["sleep", "infinity"]
