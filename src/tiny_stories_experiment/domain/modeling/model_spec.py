"""Validated sizes for a tiny decoder-only causal language model."""

from __future__ import annotations

from dataclasses import dataclass

from tiny_stories_experiment.domain.errors import InvalidModelSpecError

_MIN_VOCAB_SIZE = 2
_MIN_CONTEXT_LENGTH = 2


@dataclass(frozen=True)
class ModelSpec:
    """Sizes for one decoder-only causal language model.

    Attributes:
        vocab_size: Number of token ids the embedding and output projection share.
        embedding_size: Width of token and hidden vectors.
        layer_count: Number of causal transformer blocks.
        attention_head_count: Heads inside each block. Must divide the width.
        context_length: Maximum token positions in one row.
        feedforward_size: Hidden width of each block's position-wise layer.
    """

    vocab_size: int
    embedding_size: int
    layer_count: int
    attention_head_count: int
    context_length: int
    feedforward_size: int

    def __post_init__(self) -> None:
        """Refuse sizes that cannot form a causal transformer.

        Flow:
            1. Vocab size — require at least two token ids.
            2. Embedding size — require a positive hidden width.
            3. Layer count — require at least one transformer block.
            4. Attention heads — require a positive count that divides the width.
            5. Context length — require room for a next-token pair.
            6. Feedforward size — require a positive inner width.
        """
        # 1. Vocab size
        if self.vocab_size < _MIN_VOCAB_SIZE:
            message = f"Model vocab size {self.vocab_size} must be at least 2."
            raise InvalidModelSpecError(message)

        # 2. Embedding size
        if self.embedding_size < 1:
            message = f"Model embedding size {self.embedding_size} must be positive."
            raise InvalidModelSpecError(message)

        # 3. Layer count
        if self.layer_count < 1:
            message = f"Model layer count {self.layer_count} must be positive."
            raise InvalidModelSpecError(message)

        # 4. Attention heads
        if self.attention_head_count < 1:
            message = (
                f"Model attention head count {self.attention_head_count} "
                "must be positive."
            )
            raise InvalidModelSpecError(message)
        if self.embedding_size % self.attention_head_count != 0:
            message = (
                f"Model attention head count {self.attention_head_count} must "
                f"divide embedding size {self.embedding_size}."
            )
            raise InvalidModelSpecError(message)

        # 5. Context length
        if self.context_length < _MIN_CONTEXT_LENGTH:
            message = f"Model context length {self.context_length} must be at least 2."
            raise InvalidModelSpecError(message)

        # 6. Feedforward size
        if self.feedforward_size < 1:
            message = (
                f"Model feedforward size {self.feedforward_size} must be positive."
            )
            raise InvalidModelSpecError(message)

    @classmethod
    def tiny_debug(cls) -> ModelSpec:
        """Return the overfit-sized debug specification.

        Returns:
            A spec whose transformer stays within a few hundred thousand parameters.
        """
        return cls(
            vocab_size=64,
            embedding_size=64,
            layer_count=2,
            attention_head_count=4,
            context_length=32,
            feedforward_size=128,
        )
