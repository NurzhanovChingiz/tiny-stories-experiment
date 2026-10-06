"""Port for reading story text from one raw JSONL file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class RawStorySource(ABC):
    """Reader for story text stored in a raw JSONL file."""

    @abstractmethod
    def texts(self, path: Path) -> Iterator[str]:
        """Yield each story's text from a raw JSONL file.

        Args:
            path: Raw JSONL file to read.

        Returns:
            Story text in file order.

        Raises:
            RawStoryRecordError: A line is not an object with a string text field.
        """
