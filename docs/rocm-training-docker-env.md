# Agent instruction: recreate the ROCm training Docker environment

Use this document to build the **same GPU training environment** in a different
project. Copy the host and container contract below. Do not copy this
repository’s application code, CLI, worker, or API image.

Source of truth in the original repo: `docker/training/` plus the environment
section of `docs/future plan.md`. GPU smoke and one Lightning epoch were
confirmed on this machine on 2026-08-27.

## Goal

A Docker Compose training service on this host where:

- PyTorch is the **ROCm/HIP** build shipped in the official image (`torch.version.hip` is set).
- `torch.cuda.is_available()` is true (HIP is exposed through the `torch.cuda` API).
- Device 0 is **AMD Radeon 8060S**, architecture **`gfx1151`**.
- A small CUDA-device matmul succeeds.
- `rocminfo` lists `gfx1151` with **no** `HSA_OVERRIDE_GFX_VERSION`.

## Hardware and host (fixed)

| Item | Required value |
|------|----------------|
| Machine | AMD Ryzen AI Max+ 395 (Strix Halo) |
| GPU | Radeon 8060S, `gfx1151` (RDNA 3.5) |
| Windows driver | Adrenalin **≥ 26.2.2** |
| Guest | **WSL2**, Ubuntu **24.04** |
| GPU path | `/dev/dxg` + **ROCDXG** (`librocdxg`). Windows Adrenalin owns the GPU. |
| Docker | Docker Engine + Compose v2, available inside WSL |

This path is **not** native Linux KFD. Do not require `/dev/kfd` or `/dev/dri`.
They are absent on this WSL host.

## Do not change these decisions

- Python in the training image is **3.12**, taken from the ROCm base image. Do not use 3.14 for this image.
- Base image pin:

  ```text
  rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
  ```

  That image already contains ROCm PyTorch **2.9.1** in `/opt/venv`.
- uv pin copied into the image: `ghcr.io/astral-sh/uv:0.10.6`.
- Host ROCm installer: `amdgpu-install` **7.2.1** for Ubuntu Noble, usecase **`rocm`**, flag **`--no-dkms`**.
- Host ROCDXG pin: **librocdxg 1.2.2** (`rocdxg-roct` and `rocdxg-amd-smi-lib` debs).
- Developer machines and CI may keep a **CPU** torch from `https://download.pytorch.org/whl/cpu`. The training image must **not** install that wheel.
- Keep a separate CPU/API image if the new project has one. Do not turn the API image into the ROCm image.

Rejected approaches:

- `amdgpu-install --usecase=wsl` (invalid on amdgpu-install 7.2.x).
- Installing torch or triton from the project lock inside the training image.
- Building a custom ROCm torch, or waiting for a Python 3.14 ROCm image.
- Setting `HSA_OVERRIDE_GFX_VERSION` (including `11.0.0`). On this APU that override has caused memory-fault failures. The stack already reports `gfx1151`.

## 1. Prepare the WSL host

Run this **once on the WSL host**, not inside the container. It needs sudo.
After it succeeds, from **Windows** run `wsl --shutdown`, then reopen WSL so
the `render` and `video` groups apply.

Preconditions the script must enforce before installing:

- `/dev/dxg` exists.
- `/usr/lib/wsl/lib/libdxcore.so` is readable.
- Host is Ubuntu 24.04 (warn and continue on anything else; this recipe is tested only on 24.04).

Install steps:

1. Download:
   - `https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb`
   - `https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb`
   - `https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb`
2. `sudo dpkg -i` the amdgpu-install deb, then `sudo apt-get update -y`.
3. `sudo usermod -aG render,video "$USER"`.
4. If `rocm-core` is not already installed: `sudo amdgpu-install --usecase=rocm --no-dkms -y`.
5. `sudo dpkg -i` both librocdxg debs. If dependencies fail, `sudo apt-get -f install -y`, then `sudo ldconfig`.
6. Fail the script unless both paths exist:
   - `/opt/rocm/lib/librocdxg.so`
   - `/opt/rocm/share/rocdxg/dids.conf`
7. Append `export HSA_ENABLE_DXG_DETECTION=1` to `~/.bashrc` if it is not already there.

Host check after `wsl --shutdown` and a new shell:

```bash
export HSA_ENABLE_DXG_DETECTION=1
rocminfo | grep -E 'Name:|Marketing|gfx'
```

Expected: a GPU agent whose name/marketing line is the Radeon 8060S and whose
architecture is `gfx1151`.

## 2. Training image

Build context is the **new project root**. Dockerfile path suggestion:
`docker/training/Dockerfile`.

### uv sync rules (this is the usual failure)

The official image keeps torch in `/opt/venv`, not in system site-packages.
Point uv at that environment and **leave the image torch and triton in place**.

Required environment:

```dockerfile
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2
```

Every `uv sync` in this image must include:

```text
--frozen --inexact --no-install-package torch --no-install-package triton
```

- `--inexact` is required because the image’s ROCm torch is **not** the wheel in a CPU lockfile.
- In this original repo the lock resolves `torch==2.13.0+cpu` from the CPU index. The image torch is **2.9.1+ROCm**. Installing the lock torch replaces HIP with CPU and the GPU disappears.
- If the new project’s `pyproject.toml` indexes torch at `pytorch-cpu`, keep that for local/CI only. The container sync must still skip `torch` and `triton`.

After the project is installed, fail the image build if HIP torch was replaced:

```dockerfile
&& python -c "import torch; assert torch.version.hip, 'expected ROCm/HIP torch from /opt/venv'"
```

### Image shape

Use a multi-stage Dockerfile:

1. `FROM ghcr.io/astral-sh/uv:0.10.6 AS uv`, then `COPY --from=uv /uv /uvx /bin/`.
2. `FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.
3. Sync third-party deps with the lock mounted (`--mount=type=bind`) and a uv cache mount. Do not copy the project into the heavy-deps stage.
4. Final runtime stage copies only what training needs (project metadata, source, training config). Do not copy tests, notebooks, `.git`, or secrets.
5. Runtime user: build args `UID` and `GID` (default `1000`). The ROCm base already has uid/gid 1000 (`ubuntu`). Reuse that user when `getent` finds it. Create `app` only when the ids are absent.
6. Create and chown writable dirs before `USER`:
   - `/app/artifacts`
   - `/app/cache`
   - `/app/checkpoints`
   - `/data/datasets`
   - `/home/app/.config/miopen`
   - `/tmp`
7. Runtime env:

   ```dockerfile
   ENV HOME=/home/app \
       XDG_CACHE_HOME=/app/cache \
       UV_CACHE_DIR=/app/cache/uv \
       TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
       MIOPEN_FIND_MODE=2
   ```

8. `USER ${UID}:${GID}`. Default command can be the new project’s training worker. One-off checks use `docker compose run` or `exec`.

Match the original `.dockerignore` idea: exclude `.git`, `.venv`, caches, `data`, secrets (`.env`, keys), and notebooks. Keep `docker/training/` scripts that the image copies (smoke probe).

## 3. Compose service

File suggestion: `docker/training/docker-compose.yaml`.
Build context: repository root (`context: ../..` if the file lives in `docker/training/`).
`.env` next to the compose file (from an example, not committed secrets):

```text
UID=1000
GID=1000
SHM_SIZE=8gb
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
MIOPEN_FIND_MODE=2
```

`UID` / `GID` must match `id -u` and `id -g` on the WSL user, the image build-args, and the compose `user:` field. Otherwise mounted caches and checkpoints are not writable.

The `train` service must set all of the following. Omitting any one of the device, library mounts, or `HSA_ENABLE_DXG_DETECTION` leaves PyTorch without a GPU.

```yaml
user: "${UID:-1000}:${GID:-1000}"
environment:
  HOME: /home/app
  XDG_CACHE_HOME: /app/cache
  UV_CACHE_DIR: /app/cache/uv
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: ${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}
  MIOPEN_FIND_MODE: ${MIOPEN_FIND_MODE:-2}
  HSA_ENABLE_DXG_DETECTION: "1"
devices:
  - /dev/dxg
group_add:
  - video
  - render
cap_add:
  - SYS_PTRACE
security_opt:
  - seccomp:unconfined
ipc: host
shm_size: ${SHM_SIZE:-8gb}
volumes:
  - train_cache:/app/cache
  - train_checkpoints:/app/checkpoints
  - train_datasets:/data/datasets
  - train_miopen:/home/app/.config/miopen
  - /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro
  - /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so:ro
  - /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf:ro
```

Add the new project’s config and artifact bind mounts as needed. Do not bake secrets into the image.

Named volumes to declare: `train_cache`, `train_checkpoints`, `train_datasets`, `train_miopen`.
The MIOpen volume is required. First unique tensor shapes are slow while MIOpen compiles kernels; the cache must survive container restarts (`MIOPEN_FIND_MODE=2` reads that disk cache).

Optional, **off** unless a measured SDMA failure appears:

```text
HSA_ENABLE_SDMA=0
```

`shm_size` of 8gb (and `ipc: host`) is for DataLoader workers. AMD’s examples use 8G.

## 4. Acceptance probe

Ship a small `docker/training/smoke.py` that exits 0 only when all of these pass, and exits 1 otherwise:

1. `import torch` succeeds.
2. `torch.version.hip` is non-empty.
3. `torch.cuda.is_available()` is true.
4. `torch.cuda.get_device_name(0)` prints the Radeon device.
5. `torch.randn(64, 64, device="cuda") @` itself runs, and `.sum().item()` returns a float (no HIP memory-access fault).
6. If `rocminfo` is on `PATH`, its output contains `gfx1151`. Missing `rocminfo` is a warning, not a failure. A missing `gfx1151` string is a warning in the original probe; treat a successful device matmul as the hard gate.

Run:

```bash
cp docker/training/.env.example docker/training/.env   # set UID and GID from `id -u` / `id -g`
docker compose -f docker/training/docker-compose.yaml build train
docker compose -f docker/training/docker-compose.yaml up -d train
docker compose -f docker/training/docker-compose.yaml exec train python docker/training/smoke.py
```

Inside the container, these one-liners must agree with the script:

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.hip)'
```

Expected device string contains the Radeon 8060S name. Expected HIP version is in the **7.2** line (image is ROCm 7.2.1).

## 5. Training code constraints for the new project

- Select the GPU with Lightning `accelerator="gpu"` (or `"auto"`). Do not branch on NVIDIA-only APIs.
- Keep calling the device `"cuda"` in PyTorch. On this build that means HIP.
- Unified memory: `hipMemGetInfo` can mis-report free VRAM. Do not size batches from “90% of reported VRAM”. Set an explicit batch-size cap in config.
- Container Python must be invoked as `python`, not `uv run`. The venv is the image’s `/opt/venv`; `uv run` can fail as the non-root user.
- Dev dependency groups (ruff, mypy, pytest) stay out of the image (`UV_NO_DEV=1`).

## 6. Checklist for the other agent

- [ ] WSL Ubuntu 24.04, `/dev/dxg` and `libdxcore.so` present before host install.
- [ ] Host packages: ROCm 7.2.1 `--usecase=rocm --no-dkms`, librocdxg 1.2.2, user in `render` and `video`.
- [ ] `wsl --shutdown` done once after the host install.
- [ ] `rocminfo` on the host shows `gfx1151` with `HSA_ENABLE_DXG_DETECTION=1` and without `HSA_OVERRIDE_GFX_VERSION`.
- [ ] Image `FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.
- [ ] uv 0.10.6, `UV_PROJECT_ENVIRONMENT=/opt/venv`.
- [ ] `uv sync --frozen --inexact --no-install-package torch --no-install-package triton`.
- [ ] Image build asserts `torch.version.hip`.
- [ ] Compose passes `/dev/dxg`, the three host library mounts, `HSA_ENABLE_DXG_DETECTION=1`, `video`+`render`, `SYS_PTRACE`, `seccomp:unconfined`, `ipc: host`, `shm_size: 8gb`.
- [ ] MIOpen cache is a named volume at `/home/app/.config/miopen`.
- [ ] `smoke.py` exits 0 inside `train`.
