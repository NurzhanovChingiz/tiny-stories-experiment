"""Port for saving a trained tokenizer document."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

TOKENIZER_FILENAME = "tokenizer.json"


class TokenizerRepository(ABC):
    """Storage for one serialized tokenizer document."""

    @abstractmethod
    def save(self, destination: Path, name: str, content: bytes) -> Path:
        """Store ``tokenizer.json`` under ``destination/name`` and return its path.

        The previous document stays in place until the new bytes are ready to
        replace it. A failed write leaves that previous document unchanged.

        Args:
            destination: Directory that holds named tokenizer directories.
            name: Single path segment for this tokenizer.
            content: UTF-8 JSON document for ``tokenizer.json``.

        Returns:
            Path of the stored ``tokenizer.json``.
        """
