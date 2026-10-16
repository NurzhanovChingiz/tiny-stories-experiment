"""Text normalizer port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_story.domain.entities.normalization import NormalizeTextResult


class TextNormalizer(ABC):
    """Port that formats sentence punctuation and inserts the delimiter."""

    @abstractmethod
    def normalize(self, content: str) -> tuple[str, NormalizeTextResult]:
        """Normalize one text body.

        Args:
            content: Text to format, usually already cleaned.

        Returns:
            The normalized text and the step outcomes for this run.
        """
