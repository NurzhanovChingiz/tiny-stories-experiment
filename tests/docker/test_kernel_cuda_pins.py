"""Kernel CUDA pins stay on official PyTorch 2.14.1 CUDA 13.2."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_IMAGE = "pytorch/pytorch:2.14.1-cuda13.2-cudnn9-devel"
_TORCH_INDEX = "https://download.pytorch.org/whl/cu132"
_INDEX_NAME = "pytorch-cu132"


def test_dockerfile_uses_the_official_cuda132_devel_image() -> None:
    """The kernel image is the official 2.14.1 CUDA 13.2 devel tag."""
    text = (_ROOT / "docker" / "kernel" / "Dockerfile").read_text(encoding="utf-8")

    assert f"FROM {_KERNEL_IMAGE}" in text


def test_download_script_pulls_the_same_image() -> None:
    """download_bigfiles.sh pulls the same tag the Dockerfile builds from."""
    text = (_ROOT / "docker" / "kernel" / "download_bigfiles.sh").read_text(
        encoding="utf-8"
    )

    assert f'BASE_IMAGE="{_KERNEL_IMAGE}"' in text


def test_host_venv_installs_torch_from_cu132() -> None:
    """The host CUDA venv installs torch 2.14.1 from the cu132 index."""
    text = (_ROOT / "docker" / "kernel" / "setup_host_cuda.sh").read_text(
        encoding="utf-8"
    )

    assert f'TORCH_INDEX="{_TORCH_INDEX}"' in text
    assert 'TORCH_PIN="torch==2.14.1"' in text


def test_pyproject_names_the_cu132_index() -> None:
    """pyproject.toml names the same cu132 index the host script uses."""
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert f'name = "{_INDEX_NAME}"' in text
    assert f'url = "{_TORCH_INDEX}"' in text
    assert "cu134" not in text
    assert "cu126" not in text


def test_docs_document_the_same_pins() -> None:
    """The NVIDIA kernel doc lists the same image and wheel index."""
    text = (_ROOT / "docs" / "nvidia-kernel-docker-env.md").read_text(encoding="utf-8")

    assert _KERNEL_IMAGE in text
    assert _TORCH_INDEX in text
    assert "cu126" not in text
    assert "sudo make kernel-host" in text


def test_host_docker_script_installs_compose_and_nvidia_runtime() -> None:
    """setup_host_docker.sh installs Compose V2 and configures the NVIDIA runtime."""
    text = (_ROOT / "docker" / "kernel" / "setup_host_docker.sh").read_text(
        encoding="utf-8"
    )

    assert 'COMPOSE_PACKAGE="docker-compose-v2"' in text
    assert "nvidia-container-toolkit" in text
    assert "nvidia-ctk runtime configure --runtime=docker" in text


def test_dockerignore_keeps_kernel_smoke_in_build_context() -> None:
    """The kernel Dockerfile copies smoke.py, so dockerignore must re-include it."""
    ignore = (_ROOT / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (_ROOT / "docker" / "kernel" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY docker/kernel/smoke.py /app/smoke.py" in dockerfile
    assert "!docker/kernel/smoke.py" in ignore


def test_dockerfile_creates_venv_with_uv() -> None:
    """The official CUDA image lacks ensurepip, so the venv is created with uv."""
    text = (_ROOT / "docker" / "kernel" / "Dockerfile").read_text(encoding="utf-8")

    assert "uv venv --python python --system-site-packages /opt/venv" in text
    assert "python -m venv" not in text
