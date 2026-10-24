# NVIDIA CUDA kernel for notebooks under /app/notebooks.
# Python 3.12 and CUDA torch already live in the official image.
# Install [project] dependencies from pyproject.toml, plus the notebook and inference groups.
FROM ghcr.io/astral-sh/uv:0.10.6 AS uv

FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# Image Python is conda and has no ensurepip, so uv creates the venv.
# system-site-packages keeps image CUDA torch.
RUN uv venv --python python --system-site-packages /opt/venv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY docker/kernel/smoke.py /app/smoke.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --inexact --no-dev --group notebook --group inference --group profiling \
    && python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" \
    && python -c "import torch; assert torch.version.cuda, 'expected CUDA torch from the image'"

ARG UID=1000
ARG GID=1000

RUN mkdir -p \
        /app/cache \
        /app/profiles \
        /home/app \
        /tmp \
    && if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" app; fi \
    && if ! getent passwd "${UID}" >/dev/null; then \
         useradd --uid "${UID}" --gid "${GID}" --create-home --home-dir /home/app --shell /bin/bash app; \
       fi \
    && chown -R "${UID}:${GID}" \
        /app/cache \
        /app/profiles \
        /home/app \
        /tmp

ENV HOME=/home/app \
    XDG_CACHE_HOME=/app/cache \
    UV_CACHE_DIR=/app/cache/uv \
    PROFILE_DIR=/app/profiles

USER ${UID}:${GID}

WORKDIR /app/notebooks

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/notebooks", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True"]
