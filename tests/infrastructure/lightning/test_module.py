"""Tests for the Lightning language-model module."""

from __future__ import annotations

import pytest

from tiny_story.domain.entities.language_model import CausalLmSpec

try:
    import torch

    from tiny_story.infrastructure.lightning.module import CausalLmLightningModule
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)

_SPEC = CausalLmSpec(embedding_dim=8, hidden_size=16, num_layers=1, dropout=0.0)


def _module() -> CausalLmLightningModule:
    return CausalLmLightningModule(
        6,
        _SPEC,
        learning_rate=0.05,
        weight_decay=0.0,
        scheduler_type="cosine",
        epochs=2,
        warmup_epochs=1,
        eta_min=1e-6,
        step_size=10,
        gamma=0.1,
        max_lr=0.05,
        steps_per_epoch=1,
    )


def test_training_step_returns_a_finite_scalar_loss() -> None:
    """One integer batch yields a finite next-token loss."""
    module = _module()
    batch = torch.tensor([[2, 4, 5, 4, 3], [2, 5, 4, 5, 3]], dtype=torch.long)

    loss = module.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_configure_optimizers_returns_adamw_and_a_scheduler() -> None:
    """The module builds its own optimizer and scheduler from stored numbers."""
    configured = _module().configure_optimizers()

    assert "optimizer" in configured
    assert "lr_scheduler" in configured
