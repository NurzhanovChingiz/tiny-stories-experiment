"""Lightning adapter for a tiny decoder-only causal language model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch import LightningModule
from torch import nn

from tiny_stories_experiment.application.ports.causal_language_model import (
    CausalLanguageModel,
)
from tiny_stories_experiment.domain.modeling.model_spec import ModelSpec
from tiny_stories_experiment.domain.modeling.token_batch import validated_token_rows

if TYPE_CHECKING:
    from collections.abc import Sequence

FIXED_OVERFIT_TOKEN_IDS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5, 6, 7, 8),
    (8, 7, 6, 5, 4, 3, 2, 1),
)
OVERFIT_STEP_COUNT = 250
NEAR_ZERO_TRAINING_LOSS = 0.05
DEFAULT_LEARNING_RATE = 0.01
_EMBEDDING_INIT_STD = 0.02


class LightningCausalLanguageModel(LightningModule, CausalLanguageModel):
    """Decoder-only causal LM whose training step is the application port.

    The token embedding matrix is also the output projection. Attention is
    causal, dropout is off, and one Adam optimizer serves both ``fit_batch``
    and Lightning's trainer.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        """Assemble the decoder stack and its Adam optimizer.

        Args:
            spec: Validated causal LM sizes.
            learning_rate: Adam step size. Must be positive.

        Raises:
            ValueError: ``learning_rate`` is not positive.

        Flow:
            1. Learning rate — refuse a non-positive step size.
            2. Decoder stack — build embeddings, the causal encoder, and the norm.
            3. Optimizer — Adam over the assembled parameters.
        """
        super().__init__()
        # 1. Learning rate
        if learning_rate <= 0:
            message = f"Learning rate {learning_rate} must be positive."
            raise ValueError(message)
        self.spec = spec

        # 2. Decoder stack
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.embedding_size)
        self.position_embedding = nn.Embedding(spec.context_length, spec.embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.embedding_size,
            nhead=spec.attention_head_count,
            dim_feedforward=spec.feedforward_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spec.layer_count,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(spec.embedding_size)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=_EMBEDDING_INIT_STD)
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=_EMBEDDING_INIT_STD,
        )

        # 3. Optimizer
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def loss_on_tokens(self, token_ids: Sequence[Sequence[int]]) -> float:
        """Return the next-token loss for ``token_ids`` without updating weights.

        Args:
            token_ids: Rectangular rows of token ids.

        Returns:
            Scalar cross-entropy for the batch.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored.
        """
        with torch.no_grad():
            return float(self._next_token_loss(token_ids).item())

    def fit_batch(self, token_ids: Sequence[Sequence[int]]) -> float:
        """Take one Adam step on ``token_ids`` and return that step's loss.

        Args:
            token_ids: Rectangular rows of token ids.

        Returns:
            The loss measured before the parameter update.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored.

        Flow:
            1. Step loss — measure next-token loss in train mode.
            2. Parameter update — apply one Adam step.
            3. Scalar loss — return the measured loss as a float.
        """
        # 1. Step loss
        self.train()
        loss = self._next_token_loss(token_ids)

        # 2. Parameter update
        self._optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        self._optimizer.step()

        # 3. Scalar loss
        return float(loss.detach().item())

    def training_step(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Return the next-token loss tensor for a Lightning trainer batch.

        Args:
            *args: Positional trainer inputs. The first value is the token batch.
            **kwargs: Keyword trainer inputs. ``batch`` is read when the tensor
                is not positional.

        Returns:
            Scalar cross-entropy with a gradient.

        Raises:
            ValueError: The trainer batch is not a rank-2 tensor.
            InvalidTokenBatchError: The ids cannot be scored for this spec.

        Flow:
            1. Trainer batch — read the rank-2 token tensor from the trainer call.
            2. Next-token loss — score that batch.
        """
        # 1. Trainer batch
        candidate = args[0] if args else kwargs.get("batch")
        if not isinstance(candidate, torch.Tensor) or candidate.ndim != 2:
            message = "Lightning batch must be a rank-2 token id tensor."
            raise ValueError(message)

        # 2. Next-token loss
        rows = tuple(tuple(int(token) for token in row) for row in candidate.tolist())
        return self._next_token_loss(rows)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the Adam optimizer shared with ``fit_batch``.

        Returns:
            The module optimizer.
        """
        return self._optimizer

    def token_logits(self, token_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Return per-position vocabulary logits under a causal mask.

        Position ``t`` is computed only from tokens at positions ``0`` through
        ``t``. The output projection reuses the token embedding matrix.

        Args:
            token_ids: Candidate rows of token ids.

        Returns:
            Logits shaped ``[rows, positions, vocab]`` with a gradient.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored for this spec.

        Flow:
            1. Token rows — keep a rectangular in-vocab batch for this spec.
            2. Vocabulary logits — project the causal hidden states.
        """
        # 1. Token rows
        rows = validated_token_rows(self.spec, token_ids)

        # 2. Vocabulary logits
        return self._logits_for_rows(rows)

    def _next_token_loss(self, token_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Score each token against the next id.

        Args:
            token_ids: Candidate rows of token ids.

        Returns:
            A scalar cross-entropy tensor with a gradient.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored for this spec.

        Algorithm:
            1. Token rows — keep a rectangular in-vocab batch for this spec.
            2. Position logits — score every token under the causal mask.
            3. Shifted targets — cross-entropy of each position against the
               following id.
        """
        # 1. Token rows
        rows = validated_token_rows(self.spec, token_ids)

        # 2. Position logits
        logits = self._logits_for_rows(rows)

        # 3. Shifted targets
        tokens = torch.tensor(rows, dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.spec.vocab_size),
            tokens[:, 1:].reshape(-1),
        )

    def _logits_for_rows(self, rows: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        """Project validated token rows to vocabulary logits.

        Args:
            rows: Rectangular in-vocab token ids for this spec.

        Returns:
            Logits shaped ``[rows, positions, vocab]`` with a gradient.

        Algorithm:
            1. Token tensor — place the rows on the model device.
            2. Hidden states — add token and position embeddings, then run the
               causal stack.
            3. Vocabulary logits — project each hidden state with the tied matrix.
        """
        # 1. Token tensor
        device = self.token_embedding.weight.device
        tokens = torch.tensor(rows, dtype=torch.long, device=device)

        # 2. Hidden states
        positions = torch.arange(tokens.shape[1], device=device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1],
            device=device,
        )
        hidden = self.final_norm(
            self.encoder(hidden, mask=causal_mask, is_causal=True),
        )

        # 3. Vocabulary logits
        return nn.functional.linear(hidden, self.token_embedding.weight)


def build_tiny_debug_model(*, seed: int = 0) -> LightningCausalLanguageModel:
    """Build the tiny debug causal LM with a fixed initialization seed.

    Args:
        seed: Torch seed applied before parameter initialization.

    Returns:
        A CPU model for ``ModelSpec.tiny_debug``.
    """
    torch.manual_seed(seed)
    return LightningCausalLanguageModel(ModelSpec.tiny_debug())


def fit_token_batch(
    model: CausalLanguageModel,
    token_ids: Sequence[Sequence[int]],
    step_count: int,
) -> float:
    """Fit one batch for ``step_count`` steps and return the last loss.

    Args:
        model: Causal LM updated in place.
        token_ids: Batch reused on every step.
        step_count: Number of optimizer steps. Must be at least 1.

    Returns:
        Loss from the last step, measured before that step's update.

    Raises:
        ValueError: ``step_count`` is below 1.
        InvalidTokenBatchError: ``token_ids`` cannot be scored.

    Flow:
        1. Step budget — refuse a count below one.
        2. Repeated updates — fit the same batch once per step.
        3. Final loss — return the last measured loss.
    """
    # 1. Step budget
    if step_count < 1:
        message = f"Step count {step_count} must be at least 1."
        raise ValueError(message)

    # 2. Repeated updates
    loss = model.fit_batch(token_ids)
    for _ in range(step_count - 1):
        loss = model.fit_batch(token_ids)

    # 3. Final loss
    return loss
