"""Tests for accelerator selection."""

from __future__ import annotations

import pytest

try:
    from tiny_story.infrastructure.lightning.runtime import resolve_accelerator
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)


def test_resolve_accelerator_is_cpu_or_gpu() -> None:
    """The accelerator name is one of the two supported backends."""
    assert resolve_accelerator() in {"cpu", "gpu"}
