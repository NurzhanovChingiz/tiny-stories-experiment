"""Tiny LSTM that maps token ids to next-token logits."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from tiny_story.domain.entities.language_model import CausalLmSpec


class TinyLstmLanguageModel(nn.Module):
    """Embed tokens, run an LSTM, and project to vocabulary logits.

    Args:
        vocab_size: Output vocabulary width, including special tokens.
        spec: LSTM width settings.
        pad_id: Embedding padding index.
    """

    def __init__(
        self,
        vocab_size: int,
        spec: CausalLmSpec,
        *,
        pad_id: int,
    ) -> None:
        """Store the embedding, LSTM, dropout, and output projection.

        Args:
            vocab_size: Output vocabulary width, including special tokens.
            spec: LSTM width settings.
            pad_id: Embedding padding index.

        Returns:
            None.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            spec.embedding_dim,
            padding_idx=pad_id,
        )
        self.lstm = nn.LSTM(
            spec.embedding_dim,
            spec.hidden_size,
            num_layers=spec.num_layers,
            batch_first=True,
            dropout=spec.dropout if spec.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(spec.dropout)
        self.head = nn.Linear(spec.hidden_size, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Map a batch of token-id windows to vocabulary logits.

        Purpose:
            Predict a logit vector at every input position so the caller can
            score the next token.

        Strategy:
            Embed the ids, run the LSTM over the time axis, then project each
            hidden state to the vocabulary.

        Args:
            token_ids: Integer tensor of shape ``(batch, time)``.

        Returns:
            Float tensor of shape ``(batch, time, vocab_size)``.

        Flow:
            1. Embed the token ids.
            2. Encode the sequence with the LSTM.
            3. Project hidden states to vocabulary logits.
        """
        # 1. Embed the token ids.
        embedded = self.dropout(self.embedding(token_ids))

        # 2. Encode the sequence with the LSTM.
        hidden, _state = self.lstm(embedded)

        # 3. Project hidden states to vocabulary logits.
        return cast("Tensor", self.head(self.dropout(hidden)))
