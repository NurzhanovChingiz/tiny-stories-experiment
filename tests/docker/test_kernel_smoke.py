"""CUDA kernel smoke hard gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_SMOKE_PATH = Path(__file__).resolve().parents[2] / "docker" / "kernel" / "smoke.py"
_SPEC = importlib.util.spec_from_file_location("kernel_smoke", _SMOKE_PATH)
if _SPEC is None or _SPEC.loader is None:
    message = f"unable to load kernel smoke from {_SMOKE_PATH}"
    raise RuntimeError(message)
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules["kernel_smoke"] = smoke
_SPEC.loader.exec_module(smoke)


class _Sum:
    """Matmul reduction that yields a float."""

    def item(self) -> float:
        """Return the reduction as a Python float."""
        return 1.5


class _Tensor:
    """Minimal CUDA tensor that supports a matmul and a sum."""

    def __matmul__(self, other: object) -> _Tensor:
        """Return another tensor so the probe can call sum."""
        del other
        return _Tensor()

    def sum(self) -> _Sum:
        """Return a reduction with an item method."""
        return _Sum()


class _BrokenTensor:
    """Tensor whose matmul raises a runtime error."""

    def __matmul__(self, other: object) -> _Tensor:
        """Raise the device fault the probe must surface."""
        del other
        message = "CUDA error: device-side assert triggered"
        raise RuntimeError(message)


def _torch_module(
    *,
    cuda_version: str | None = "13.2",
    available: bool = True,
    device_name: str = "NVIDIA GeForce RTX 4090",
    tensor: object | None = None,
) -> SimpleNamespace:
    """Build a torch stand-in for one probe case.

    Args:
        cuda_version: Value of ``torch.version.cuda``.
        available: Value of ``torch.cuda.is_available()``.
        device_name: Value of ``torch.cuda.get_device_name(0)``.
        tensor: Object returned by ``randn``.

    Returns:
        An object with the torch attributes the probe reads.
    """
    sample = _Tensor() if tensor is None else tensor

    def _randn(*_size: int, **_kwargs: object) -> object:
        return sample

    return SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda _index: device_name,
        ),
        randn=_randn,
    )


def test_evaluate_probe_accepts_any_cuda_device_name() -> None:
    """A named NVIDIA card is enough; the probe does not require one model."""
    facts = smoke.ProbeFacts(
        import_error="",
        cuda_build=True,
        cuda_available=True,
        device_name="Tesla T4",
        matmul_error="",
        nvidia_smi_text="NVIDIA-SMI",
    )

    decision = smoke.evaluate_probe(facts)

    assert decision.ok
    assert decision.errors == ()
    assert decision.warnings == ()


def test_evaluate_probe_warns_when_nvidia_smi_is_missing() -> None:
    """A missing nvidia-smi binary is a warning when the torch gates pass."""
    facts = smoke.ProbeFacts(
        import_error="",
        cuda_build=True,
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 4090",
        matmul_error="",
        nvidia_smi_text=None,
    )

    decision = smoke.evaluate_probe(facts)

    assert decision.ok
    assert decision.warnings == ("nvidia-smi is not on PATH",)


def test_evaluate_probe_rejects_unavailable_cuda() -> None:
    """A CUDA build without a visible device is a hard failure."""
    facts = smoke.ProbeFacts(
        import_error="",
        cuda_build=True,
        cuda_available=False,
        device_name="",
        matmul_error="",
        nvidia_smi_text=None,
    )

    decision = smoke.evaluate_probe(facts)

    assert not decision.ok
    assert decision.errors == ("torch.cuda.is_available() is false",)


def test_evaluate_probe_rejects_an_empty_device_name() -> None:
    """A visible CUDA device with no name is a hard failure."""
    facts = smoke.ProbeFacts(
        import_error="",
        cuda_build=True,
        cuda_available=True,
        device_name="",
        matmul_error="",
        nvidia_smi_text="nvidia-smi",
    )

    decision = smoke.evaluate_probe(facts)

    assert not decision.ok
    assert decision.errors == ("CUDA device 0 has an empty name",)


def test_evaluate_probe_rejects_a_cpu_torch_build() -> None:
    """A torch wheel without CUDA is a hard failure."""
    facts = smoke.ProbeFacts(
        import_error="",
        cuda_build=False,
        cuda_available=False,
        device_name="",
        matmul_error="",
        nvidia_smi_text=None,
    )

    decision = smoke.evaluate_probe(facts)

    assert not decision.ok
    assert decision.errors == ("torch.version.cuda is empty; expected a CUDA build",)


def test_evaluate_probe_rejects_an_import_error_without_extra_cuda_gates() -> None:
    """An import failure is the only torch error when torch is missing."""
    facts = smoke.ProbeFacts(
        import_error="No module named 'torch'",
        cuda_build=False,
        cuda_available=False,
        device_name="",
        matmul_error="",
        nvidia_smi_text=None,
    )

    decision = smoke.evaluate_probe(facts)

    assert not decision.ok
    assert decision.errors == ("torch import failed: No module named 'torch'",)


def test_read_device_records_cuda_facts_from_torch() -> None:
    """Device read copies CUDA build, availability, and the device-0 name."""
    facts = smoke.read_device(_torch_module(), "")

    assert facts == smoke.DeviceFacts(
        import_error="",
        cuda_build=True,
        cuda_available=True,
        device_name="NVIDIA GeForce RTX 4090",
    )


def test_run_device_matmul_returns_empty_on_success() -> None:
    """A 64 by 64 CUDA matmul that reduces to a float is success."""
    assert smoke.run_device_matmul(_torch_module()) == ""


def test_run_device_matmul_returns_the_runtime_error() -> None:
    """A device fault during matmul is returned as text."""
    error = smoke.run_device_matmul(_torch_module(tensor=_BrokenTensor()))

    assert "device-side assert" in error


def test_main_writes_success_when_the_probe_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prints a pass line and exits 0 when hard gates succeed."""
    monkeypatch.setattr(
        smoke,
        "probe",
        lambda: smoke.ProbeDecision(ok=True, errors=(), warnings=()),
    )

    assert smoke.main() == 0
    assert capsys.readouterr().out == "cuda probe passed\n"


def test_read_nvidia_smi_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing nvidia-smi binary is reported as None."""
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: None)

    assert smoke.read_nvidia_smi() is None


def test_read_nvidia_smi_returns_a_marker_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present nvidia-smi binary is reported as a marker, not output text."""
    monkeypatch.setattr(smoke.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    assert smoke.read_nvidia_smi() == "nvidia-smi"


def test_import_torch_keeps_the_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing torch module becomes an import error string."""

    def _missing_torch(name: str) -> ModuleType:
        if name == "torch":
            message = "No module named 'torch'"
            raise ImportError(message)
        return sys.modules[name]

    monkeypatch.setattr(smoke.importlib, "import_module", _missing_torch)
    module, error = smoke.import_torch()

    assert module is None
    assert error == "No module named 'torch'"
