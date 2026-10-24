"""Tests for Lightning scheduler construction."""

from __future__ import annotations

import pytest

try:
    import torch

    from tiny_story.infrastructure.lightning.scheduler import build_scheduler
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)


def test_build_scheduler_returns_epoch_cosine_warmup() -> None:
    """cosine_warmup wraps a sequential scheduler on epoch interval."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    config = build_scheduler(
        optimizer,
        scheduler_type="cosine_warmup",
        epochs=4,
        warmup_epochs=1,
        eta_min=1e-6,
        step_size=10,
        gamma=0.1,
        max_lr=0.1,
    )

    assert config["interval"] == "epoch"
    assert config["frequency"] == 1
    assert config["scheduler"] is not None


def test_build_scheduler_requires_steps_for_onecycle() -> None:
    """OneCycleLR fails closed without steps_per_epoch."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    with pytest.raises(ValueError, match="steps_per_epoch"):
        build_scheduler(
            optimizer,
            scheduler_type="onecycle",
            epochs=2,
            warmup_epochs=1,
            eta_min=1e-6,
            step_size=10,
            gamma=0.1,
            max_lr=0.1,
        )


def test_build_scheduler_rejects_an_unknown_name() -> None:
    """An unknown scheduler name is rejected."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    with pytest.raises(ValueError, match="Unknown scheduler_type"):
        build_scheduler(
            optimizer,
            scheduler_type="mystery",
            epochs=2,
            warmup_epochs=1,
            eta_min=1e-6,
            step_size=10,
            gamma=0.1,
            max_lr=0.1,
        )
