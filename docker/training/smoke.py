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
    """
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        empty = TorchDeviceFacts(
            hip_version="",
            cuda_available=False,
            device_name="",
        )
        return str(exc), empty, None

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
    """
    try:
        sample = torch_module.randn(64, 64, device="cuda")
        total = (sample @ sample).sum().item()
    except (AttributeError, RuntimeError, TypeError) as exc:
        return None, str(exc)
    if isinstance(total, float):
        return total, None
    return None, "matmul sum was not a float"


def read_rocminfo() -> str | None:
    """Read rocminfo output when the binary is on PATH.

    Returns:
        Combined stdout and stderr, or None when rocminfo is absent.
    """
    binary = shutil.which("rocminfo")
    if binary is None:
        return None
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
    """
    warnings: list[str] = []
    if facts.rocminfo_text is None:
        warnings.append("rocminfo is not on PATH")
    elif _ARCH_MARK not in facts.rocminfo_text:
        warnings.append("rocminfo output does not contain gfx1151")

    passed = (
        facts.torch_error is None
        and facts.hip_version != ""
        and facts.cuda_available
        and _DEVICE_MARK in facts.device_name
        and facts.matmul_error is None
        and isinstance(facts.matmul_sum, float)
    )
    if passed:
        return 0, tuple(warnings)
    return 1, tuple(warnings)


def main() -> int:
    """Decide whether this container's HIP GPU can run a device matmul.

    Strategy: collect torch, matmul, and rocminfo facts, then apply the hard
    gates. A missing rocminfo binary or a missing gfx1151 string is a warning.

    Flow:
        1. Read the HIP torch device facts.
        2. Run a 64 by 64 CUDA-device matmul when CUDA is available.
        3. Read rocminfo output when the binary is on PATH.
        4. Exit from the hard gates and warn about rocminfo.

    Returns:
        ``0`` when HIP torch, the Radeon device, and the matmul succeed.
    """
    # 1. Read the HIP torch device facts.
    torch_error, device, torch_module = read_torch_device()
    # 2. Run a 64 by 64 CUDA-device matmul when CUDA is available.
    if torch_module is not None and device.cuda_available:
        matmul_sum, matmul_error = run_device_matmul(torch_module)
    else:
        matmul_sum, matmul_error = None, None
    # 3. Read rocminfo output when the binary is on PATH.
    rocminfo_text = read_rocminfo()
    facts = ProbeFacts(
        torch_error=torch_error,
        hip_version=device.hip_version,
        cuda_available=device.cuda_available,
        device_name=device.device_name,
        matmul_sum=matmul_sum,
        matmul_error=matmul_error,
        rocminfo_text=rocminfo_text,
    )
    # 4. Exit from the hard gates and warn about rocminfo.
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
