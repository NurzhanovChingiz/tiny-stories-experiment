"""Acceptance probe for the NVIDIA CUDA kernel environment."""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class DeviceFacts:
    """CUDA-device facts read from the environment torch.

    Args:
        import_error: Import failure text, or empty when torch imported.
        cuda_build: Whether ``torch.version.cuda`` was set.
        cuda_available: Whether ``torch.cuda.is_available()`` was true.
        device_name: Device 0 name, or empty when CUDA is unavailable.
    """

    import_error: str
    cuda_build: bool
    cuda_available: bool
    device_name: str


@dataclass(frozen=True)
class ProbeFacts:
    """Combined torch, matmul, and nvidia-smi observations.

    Args:
        import_error: Import failure text, or empty when torch imported.
        cuda_build: Whether this torch was a CUDA build.
        cuda_available: Whether CUDA was available.
        device_name: Device 0 name, or empty when CUDA is unavailable.
        matmul_error: Matmul failure text, or empty on success.
        nvidia_smi_text: Present marker when nvidia-smi is on PATH, or None.
    """

    import_error: str
    cuda_build: bool
    cuda_available: bool
    device_name: str
    matmul_error: str
    nvidia_smi_text: str | None


@dataclass(frozen=True)
class ProbeDecision:
    """Hard-gate result for the CUDA probe.

    Args:
        ok: Whether the hard gates passed.
        errors: Blocking failure messages.
        warnings: Non-blocking notices.
    """

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def import_torch() -> tuple[ModuleType | None, str]:
    """Import torch from the active environment.

    Returns:
        The imported module and an empty error, or None and the import error.
    """
    try:
        module = importlib.import_module("torch")
    except ImportError as exc:
        return None, str(exc)
    return module, ""


def read_device(torch_module: object | None, import_error: str) -> DeviceFacts:
    """Read CUDA build and device facts from the imported torch.

    Args:
        torch_module: Imported torch module, or None after an import error.
        import_error: Import failure text, or empty when torch imported.

    Returns:
        CUDA build, availability, and device-0 name.

    Flow:
        1. Keep the import error when torch is missing.
        2. Read the CUDA version, availability, and device name.
    """
    # 1. Keep the import error when torch is missing.
    if torch_module is None:
        return DeviceFacts(
            import_error=import_error,
            cuda_build=False,
            cuda_available=False,
            device_name="",
        )

    # 2. Read the CUDA version, availability, and device name.
    version = getattr(torch_module, "version", None)
    cuda_build = bool(getattr(version, "cuda", None))
    cuda = getattr(torch_module, "cuda", None)
    available = bool(cuda.is_available()) if cuda is not None else False
    device_name = ""
    if available and cuda is not None:
        device_name = str(cuda.get_device_name(0))
    return DeviceFacts(
        import_error="",
        cuda_build=cuda_build,
        cuda_available=available,
        device_name=device_name,
    )


def run_device_matmul(torch_module: object) -> str:
    """Run one 64 by 64 matmul on the CUDA device.

    Args:
        torch_module: Imported torch module from the CUDA environment.

    Returns:
        An empty string on success, or the matmul error text.

    Flow:
        1. Device matmul — multiply a 64 by 64 CUDA sample, or return the error.
    """
    # 1. Device matmul — multiply a 64 by 64 CUDA sample, or return the error.
    randn = getattr(torch_module, "randn", None)
    if not callable(randn):
        return "torch.randn is missing"
    try:
        sample = randn(64, 64, device="cuda")
        result = sample @ sample
        sum_method = getattr(result, "sum", None)
        if sum_method is None:
            return "matmul result has no sum"
        numeric = sum_method()
        item = getattr(numeric, "item", None)
        if item is None:
            return "matmul sum has no item"
        float(item())
    except (RuntimeError, TypeError, ValueError) as exc:
        return str(exc)
    return ""


def read_nvidia_smi() -> str | None:
    """Return a present marker when nvidia-smi is on PATH.

    Returns:
        A non-empty marker when the binary exists, or None when it is absent.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    return "nvidia-smi"


def evaluate_probe(facts: ProbeFacts) -> ProbeDecision:
    """Apply hard gates and warnings to collected probe facts.

    A missing nvidia-smi binary is a warning. Any CUDA device name is accepted.

    Args:
        facts: Collected torch, matmul, and nvidia-smi observations.

    Returns:
        Whether the hard gates passed, plus errors and warnings.

    Flow:
        1. Collect warnings for a missing nvidia-smi binary.
        2. Hard gates — require torch, a CUDA build, a visible GPU, and a float matmul.
    """
    # 1. Collect warnings for a missing nvidia-smi binary.
    warnings: list[str] = []
    if facts.nvidia_smi_text is None:
        warnings.append("nvidia-smi is not on PATH")

    # 2. Hard gates — require torch, a CUDA build, a visible GPU, and a float matmul.
    errors: list[str] = []
    if facts.import_error:
        errors.append(f"torch import failed: {facts.import_error}")
    elif not facts.cuda_build:
        errors.append("torch.version.cuda is empty; expected a CUDA build")
    elif not facts.cuda_available:
        errors.append("torch.cuda.is_available() is false")
    elif not facts.device_name:
        errors.append("CUDA device 0 has an empty name")
    if facts.matmul_error:
        errors.append(f"device matmul failed: {facts.matmul_error}")
    return ProbeDecision(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def probe() -> ProbeDecision:
    """Decide whether this environment's CUDA GPU can run a device matmul.

    Strategy: collect torch, matmul, and nvidia-smi facts, then apply the hard
    gates. A missing nvidia-smi binary is a warning. Any CUDA device is accepted.

    Returns:
        The hard-gate decision for this environment.

    Flow:
        1. Device read — read the CUDA torch device, or keep the import error.
        2. Device matmul — run a 64 by 64 CUDA matmul when CUDA is available.
        3. nvidia-smi text — read nvidia-smi when the binary is on PATH.
        4. Probe facts — combine the torch, matmul, and nvidia-smi observations.
        5. Evaluate — apply the hard gates and warnings.
    """
    # 1. Device read — read the CUDA torch device, or keep the import error.
    torch_module, import_error = import_torch()
    device = read_device(torch_module, import_error)

    # 2. Device matmul — run a 64 by 64 CUDA matmul when CUDA is available.
    matmul_error = ""
    if torch_module is not None and device.cuda_available:
        matmul_error = run_device_matmul(torch_module)

    # 3. nvidia-smi text — read nvidia-smi when the binary is on PATH.
    nvidia_smi_text = read_nvidia_smi()

    # 4. Probe facts — combine the torch, matmul, and nvidia-smi observations.
    facts = ProbeFacts(
        import_error=device.import_error,
        cuda_build=device.cuda_build,
        cuda_available=device.cuda_available,
        device_name=device.device_name,
        matmul_error=matmul_error,
        nvidia_smi_text=nvidia_smi_text,
    )

    # 5. Evaluate — apply the hard gates and warnings.
    return evaluate_probe(facts)


def main() -> int:
    """Print probe results and return the process exit code.

    Returns:
        ``0`` when CUDA torch, a visible GPU, and the matmul succeed.
    """
    decision = probe()
    for warning in decision.warnings:
        sys.stderr.write(f"warning: {warning}\n")
    for error in decision.errors:
        sys.stderr.write(f"error: {error}\n")
    if decision.ok:
        sys.stdout.write("cuda probe passed\n")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
