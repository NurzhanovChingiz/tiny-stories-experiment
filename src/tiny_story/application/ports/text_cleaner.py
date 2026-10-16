"""Text cleaner port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiny_story.domain.entities.cleaning import CleanTextResult


class TextCleaner(ABC):
    """Port that removes junk and maps text onto the allowed characters."""

    @abstractmethod
    def clean(self, content: str) -> tuple[str, CleanTextResult]:
        """Clean one text body.

        Args:
            content: Raw text content.

        Returns:
            The cleaned text and the step outcomes for this run.
        """
