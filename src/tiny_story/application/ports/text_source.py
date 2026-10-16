"""TextSource gateway — reads a file into text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_story.domain.entities.text import Text


class TextSource(ABC):
    """Port for reading one file as text values."""

    @abstractmethod
    def read(self, path: Path) -> tuple[Text, ...]:
        """Read a file and return its text values.

        Args:
            path: File to read.

        Returns:
            The text values stored in the file, in file order.
        """
