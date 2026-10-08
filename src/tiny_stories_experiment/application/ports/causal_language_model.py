"""Port for one causal language-model training step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class CausalLanguageModel(ABC):
    """Next-token model that reports loss and can take one optimizer step."""

    @abstractmethod
    def loss_on_tokens(self, token_ids: Sequence[Sequence[int]]) -> float:
        """Return the next-token loss for ``token_ids`` without updating weights.

        Args:
            token_ids: Rectangular rows of token ids.

        Returns:
            A finite scalar cross-entropy when the batch is legal.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored.
        """

    @abstractmethod
    def fit_batch(self, token_ids: Sequence[Sequence[int]]) -> float:
        """Update parameters on ``token_ids`` and return that step's loss.

        The returned value is the loss measured before the parameter update.

        Args:
            token_ids: Rectangular rows of token ids.

        Returns:
            Scalar next-token loss for this step.

        Raises:
            InvalidTokenBatchError: ``token_ids`` cannot be scored.
        """
