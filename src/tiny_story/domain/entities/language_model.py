"""Causal language-model size settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CausalLmSpec:
    """Width of one LSTM language model, without a vocabulary size.

    Attributes:
        embedding_dim: Token embedding width.
        hidden_size: LSTM hidden state width.
        num_layers: Stacked LSTM layers.
        dropout: Dropout probability applied to embeddings and the output.
    """

    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
