"""Text tokenizer port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_story.domain.entities.tokenization import TokenizeTextResult


class TextTokenizer(ABC):
    """Port that turns story text into integer token ids."""

    @abstractmethod
    def tokenize(self, content: str) -> tuple[tuple[int, ...], TokenizeTextResult]:
        """Tokenize one text body.

        Args:
            content: Text to split, usually already normalized.

        Returns:
            The token ids and the count for this run.
        """
