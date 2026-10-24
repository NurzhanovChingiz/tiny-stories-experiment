# NVIDIA CUDA kernel environment

The Jupyter kernel and the optional host venv use official CUDA PyTorch. AMD ROCm, `/dev/dxg`, and HIP are not part of this checkout.

## Pins

- Container image: `pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel`
- Host CUDA wheel index: `https://download.pytorch.org/whl/cu132`
- Host torch pin: `torch==2.14.1`
- Python: 3.12
- uv image: `ghcr.io/astral-sh/uv:0.10.6`

`2.14.1-cuda13.2-cudnn9-devel` is the newest official CUDA tag for PyTorch 2.14.1 on Docker Hub. CUDA 13.4 is NVIDIA toolkit / PyTorch 2.15 experimental only; there is no `pytorch/pytorch:2.14.1-cuda13.4-*` tag. The compose file does not use a floating `latest` tag.

## Host prerequisites

1. A recent NVIDIA Windows driver with WSL GPU support.
2. `nvidia-smi` works inside WSL (`nvidia-smi -L` lists at least one GPU).
3. Docker Engine plus NVIDIA Container Toolkit, so `docker compose` can pass `gpus: all`.

On Ubuntu 24.04 WSL, `docker.io` does not ship the Compose plugin. Install Compose V2, add your user to the `docker` group, and configure the NVIDIA runtime with:

```bash
sudo make kernel-host
```

Open a new shell after that so the `docker` group applies, or run `sg docker -c 'make kernel'`. This repository does not vendor NVIDIA host packages. The Windows driver installs the WSL GPU user-mode stack. Do not install a Linux NVIDIA kernel driver inside WSL.

## Container

```bash
sudo make kernel-host
./docker/kernel/download_bigfiles.sh
make kernel
make kernel-smoke
```

`download_bigfiles.sh` writes `docker/kernel/.env` from the example and pulls the pinned images. `make kernel` builds `docker/kernel/Dockerfile` and starts Jupyter on `127.0.0.1:8888`.

The image keeps official CUDA torch. `uv sync` is `--inexact` and `pyproject.toml` excludes `torch`, so the lockfile cannot replace the GPU wheel.

`make kernel-smoke` runs `docker/kernel/smoke.py` inside the container. Hard gates:

1. torch imports
2. `torch.version.cuda` is set
3. `torch.cuda.is_available()` is true
4. device 0 has a non-empty name (any NVIDIA GPU)
5. a 64 by 64 CUDA matmul reduces to a float

A missing `nvidia-smi` binary is a warning.

## Host CUDA venv

`pyproject.toml` still excludes torch. Host GPU work uses a second environment:

```bash
make host-cuda
.venv-cuda/bin/python docker/kernel/smoke.py
```

That target creates `.venv-cuda`, syncs the notebook and inference groups, then installs `torch==2.14.1` from the cu132 index. Do not `uv add torch` to the project. Do not install a CPU torch wheel into the kernel container.

## Notebook

`notebooks/00_environment/gpu_pytorch.ipynb` checks that the kernel can see CUDA. Use the container Python at `/opt/venv`. Do not `uv run` inside that container.
