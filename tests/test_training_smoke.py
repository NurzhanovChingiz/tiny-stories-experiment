"""Contract tests for the ROCm training acceptance decision."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_READY = {
    "torch_error": None,
    "hip_version": "7.2.1",
    "cuda_available": True,
    "device_name": "AMD Radeon 8060S Graphics",
    "matmul_sum": 1.0,
    "matmul_error": None,
    "rocminfo_text": "Name: gfx1151\n",
}


def _load_smoke() -> Any:
    path = Path(__file__).resolve().parents[1] / "docker" / "training" / "smoke.py"
    spec = importlib.util.spec_from_file_location("training_smoke", path)
    if spec is None or spec.loader is None:
        message = f"could not load {path}"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke() -> Any:
    """Load the container probe without installing it as a package.

    Returns:
        The loaded smoke module.
    """
    return _load_smoke()


def test_ready_probe_exits_zero(smoke: Any) -> None:
    """A HIP device matmul on the Radeon name exits 0 with no warnings."""
    facts = smoke.ProbeFacts(**_READY)
    status, warnings = smoke.acceptance_status(facts)
    assert status == 0
    assert warnings == ()


def test_missing_matmul_exits_one(smoke: Any) -> None:
    """HIP and a device name do not pass when the matmul did not finish."""
    facts = smoke.ProbeFacts(**{**_READY, "matmul_sum": None, "matmul_error": "fault"})
    status, _warnings = smoke.acceptance_status(facts)
    assert status == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch_error", "import torch failed"),
        ("hip_version", ""),
        ("cuda_available", False),
        ("device_name", "AMD Radeon Graphics"),
    ],
)
def test_hard_gate_failure_exits_one(smoke: Any, field: str, value: object) -> None:
    """Each hard gate fails closed on its own."""
    facts = smoke.ProbeFacts(**{**_READY, field: value})
    status, _warnings = smoke.acceptance_status(facts)
    assert status == 1


@pytest.mark.parametrize(
    ("rocminfo_text", "warning"),
    [
        (None, "rocminfo is not on PATH"),
        ("Marketing Name: AMD Radeon\n", "rocminfo output does not contain gfx1151"),
    ],
)
def test_rocminfo_gaps_warn_without_failing(
    smoke: Any,
    rocminfo_text: str | None,
    warning: str,
) -> None:
    """Rocminfo problems stay warnings when the device matmul succeeded."""
    facts = smoke.ProbeFacts(**{**_READY, "rocminfo_text": rocminfo_text})
    status, warnings = smoke.acceptance_status(facts)
    assert status == 0
    assert warnings == (warning,)
