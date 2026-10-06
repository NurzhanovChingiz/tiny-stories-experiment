"""Acceptance probe for the ROCm training container."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

_DEVICE_MARK = "8060S"
_ARCH_MARK = "gfx1151"


@dataclass(frozen=True)
class TorchDeviceFacts:
    """HIP and CUDA-device facts read from the image torch.

    Attributes:
        hip_version: ``torch.version.hip`` text, or empty when unset.
        cuda_available: Whether ``torch.cuda.is_available()`` was true.
        device_name: Device 0 name, or empty when CUDA is unavailable.
    """

    hip_version: str
    cuda_available: bool
    device_name: str


@dataclass(frozen=True)
class ProbeFacts:
    """Observations the acceptance decision needs.

    Attributes:
        torch_error: Torch import or device-read failure, or None.
        hip_version: ``torch.version.hip`` text.
        cuda_available: Whether CUDA (HIP) was available.
        device_name: Device 0 name, or empty.
        matmul_sum: Sum of the device matmul, or None when it did not finish.
        matmul_error: Matmul failure text, or None.
        rocminfo_text: Combined rocminfo output, or None when it is missing.
    """

    torch_error: str | None
    hip_version: str
    cuda_available: bool
    device_name: str
    matmul_sum: float | None
    matmul_error: str | None
    rocminfo_text: str | None


def read_torch_device() -> tuple[str | None, TorchDeviceFacts, Any]:
    """Import the image torch and read its HIP device.

    Returns:
        An error string or None, the device facts, and the torch module or None.

    Flow:
        1. Torch import — load torch, or return empty facts when import fails.
        2. Device facts — read the HIP version, CUDA availability, and device name.
    """
    # 1. Torch import
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        empty = TorchDeviceFacts(
            hip_version="",
            cuda_available=False,
            device_name="",
        )
        return str(exc), empty, None

    # 2. Device facts
    version = getattr(torch_module, "version", None)
    hip_value = getattr(version, "hip", None)
    hip_version = hip_value if isinstance(hip_value, str) else ""
    cuda = getattr(torch_module, "cuda", None)
    available = bool(cuda.is_available()) if cuda is not None else False
    device_name = ""
    if available and cuda is not None:
        device_name = str(cuda.get_device_name(0))
    facts = TorchDeviceFacts(hip_version, available, device_name)
    return None, facts, torch_module


def run_device_matmul(torch_module: Any) -> tuple[float | None, str | None]:
    """Run one 64 by 64 matmul on the CUDA device.

    Args:
        torch_module: Imported torch module from the ROCm image.

    Returns:
        The matmul sum and None, or None and an error string.

    Flow:
        1. Device matmul — multiply a 64 by 64 CUDA sample, or return the error.
        2. Float sum — return the sum when it is a float.
    """
    # 1. Device matmul
    try:
        sample = torch_module.randn(64, 64, device="cuda")
        total = (sample @ sample).sum().item()
    except (AttributeError, RuntimeError, TypeError) as exc:
        return None, str(exc)
    # 2. Float sum
    if isinstance(total, float):
        return total, None
    return None, "matmul sum was not a float"


def read_rocminfo() -> str | None:
    """Read rocminfo output when the binary is on PATH.

    Returns:
        Combined stdout and stderr, or None when rocminfo is absent.

    Flow:
        1. Binary lookup — return None when rocminfo is not on PATH.
        2. Combined output — return stdout and stderr from that binary.
    """
    # 1. Binary lookup
    binary = shutil.which("rocminfo")
    if binary is None:
        return None
    # 2. Combined output
    completed = subprocess.run(
        [binary],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def acceptance_status(facts: ProbeFacts) -> tuple[int, tuple[str, ...]]:
    """Decide the probe exit code from the hard gates.

    A missing rocminfo binary, or rocminfo text without gfx1151, is a warning.
    The device matmul remains the hard gate.

    Args:
        facts: Collected torch, matmul, and rocminfo observations.

    Returns:
        Exit code and warning lines. Exit code 0 means the hard gates passed.

    Flow:
        1. Rocminfo warnings — note a missing binary or a missing gfx1151 string.
        2. Hard gates — require torch, HIP, the Radeon name, and a float matmul.
        3. Exit code — return 0 when the gates pass, otherwise 1.
    """
    # 1. Rocminfo warnings
    warnings: list[str] = []
    if facts.rocminfo_text is None:
        warnings.append("rocminfo is not on PATH")
    elif _ARCH_MARK not in facts.rocminfo_text:
        warnings.append("rocminfo output does not contain gfx1151")

    # 2. Hard gates
    passed = (
        facts.torch_error is None
        and facts.hip_version != ""
        and facts.cuda_available
        and _DEVICE_MARK in facts.device_name
        and facts.matmul_error is None
        and isinstance(facts.matmul_sum, float)
    )
    # 3. Exit code
    if passed:
        return 0, tuple(warnings)
    return 1, tuple(warnings)


def main() -> int:
    """Decide whether this container's HIP GPU can run a device matmul.

    Strategy: collect torch, matmul, and rocminfo facts, then apply the hard
    gates. A missing rocminfo binary or a missing gfx1151 string is a warning.

    Returns:
        ``0`` when HIP torch, the Radeon device, and the matmul succeed.

    Flow:
        1. Device read — read the HIP torch device, or keep the import error.
        2. Device matmul — run a 64 by 64 CUDA matmul when CUDA is available.
        3. Rocminfo text — read rocminfo when the binary is on PATH.
        4. Probe facts — combine the torch, matmul, and rocminfo observations.
        5. Probe result — exit from the hard gates and write warnings.
    """
    # 1. Device read
    torch_error, device, torch_module = read_torch_device()
    # 2. Device matmul
    if torch_module is not None and device.cuda_available:
        matmul_sum, matmul_error = run_device_matmul(torch_module)
    else:
        matmul_sum, matmul_error = None, None
    # 3. Rocminfo text
    rocminfo_text = read_rocminfo()
    # 4. Probe facts
    facts = ProbeFacts(
        torch_error=torch_error,
        hip_version=device.hip_version,
        cuda_available=device.cuda_available,
        device_name=device.device_name,
        matmul_sum=matmul_sum,
        matmul_error=matmul_error,
        rocminfo_text=rocminfo_text,
    )
    # 5. Probe result
    status, warnings = acceptance_status(facts)
    if device.device_name:
        sys.stdout.write(f"{device.device_name}\n")
    for warning in warnings:
        sys.stderr.write(f"warning: {warning}\n")
    if torch_error is not None:
        sys.stderr.write(f"{torch_error}\n")
    if matmul_error is not None:
        sys.stderr.write(f"{matmul_error}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
