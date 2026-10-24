"""Tests for the tiny LSTM language model."""

from __future__ import annotations

import pytest

from tiny_story.domain.entities.language_model import CausalLmSpec

try:
    import torch

    from tiny_story.infrastructure.lightning.lstm import TinyLstmLanguageModel
except ImportError:
    pytest.skip("torch is required", allow_module_level=True)


def test_lstm_forward_returns_logits_for_each_position() -> None:
    """A batch of token windows maps to vocabulary logits at every step."""
    model = TinyLstmLanguageModel(
        6,
        CausalLmSpec(embedding_dim=8, hidden_size=16, num_layers=1, dropout=0.0),
        pad_id=0,
    )
    token_ids = torch.tensor([[2, 4, 5, 3], [2, 4, 4, 3]], dtype=torch.long)

    logits = model(token_ids)

    assert logits.shape == (2, 4, 6)
    assert torch.isfinite(logits).all()
