"""Overfit-one-batch check for the LSTM Lightning module."""

from __future__ import annotations

import pytest

from tiny_story.domain.entities.language_model import CausalLmSpec

try:
    import torch

    from tiny_story.infrastructure.lightning.module import CausalLmLightningModule
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)


def test_repeated_steps_on_one_batch_reduce_loss() -> None:
    """The same token batch should move toward lower next-token loss."""
    module = CausalLmLightningModule(
        6,
        CausalLmSpec(embedding_dim=16, hidden_size=32, num_layers=1, dropout=0.0),
        learning_rate=0.2,
        weight_decay=0.0,
        scheduler_type="cosine",
        epochs=1,
        warmup_epochs=0,
        eta_min=1e-6,
        step_size=10,
        gamma=0.1,
        max_lr=0.2,
        steps_per_epoch=1,
    )
    batch = torch.tensor([[2, 4, 5, 4, 5, 3]], dtype=torch.long)
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    first_loss = module.training_step(batch, 0)
    first_value = float(first_loss.detach())
    loss = first_loss
    for _ in range(40):
        optimizer.zero_grad()
        loss = module.training_step(batch, 0)
        loss.backward()
        optimizer.step()
    last_value = float(loss.detach())

    assert first_value > 0
    assert last_value < first_value
