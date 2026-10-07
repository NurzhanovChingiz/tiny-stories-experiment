"""Port for training a tokenizer from story text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tiny_stories_experiment.application.results.trained_tokenizer import (
        TrainedTokenizer,
    )
    from tiny_stories_experiment.domain.tokenization.tokenizer_spec import (
        TokenizerSpec,
    )


class TokenizerTrainer(ABC):
    """Train a tokenizer from an ordered stream of stories."""

    @abstractmethod
    def train(self, spec: TokenizerSpec, texts: Iterable[str]) -> TrainedTokenizer:
        """Train on ``texts`` and return the serialized tokenizer.

        Args:
            spec: Vocab size, minimum pair frequency, and special tokens.
            texts: Story text in read order. The trainer consumes this once.

        Returns:
            The serialized tokenizer and how many stories were read.

        Raises:
            TokenizerTrainingError: The spec cannot build this trainer's
                vocabulary, or the stream contains no stories.
        """
