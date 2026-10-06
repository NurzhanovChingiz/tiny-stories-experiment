"""Port for writing derived story text for one split."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit


def processed_filename(split: DatasetSplit) -> str:
    """Return the derived JSONL name for a split.

    Args:
        split: Published split whose stories are being written.

    Returns:
        ``train.jsonl`` or ``valid.jsonl``.
    """
    return f"{split.value}.jsonl"


class ProcessedTextStore(ABC):
    """Storage for derived story text, one JSONL file per split."""

    @abstractmethod
    def prepare(self, destination: Path) -> None:
        """Create the processed directory when it is missing.

        Args:
            destination: Directory that will hold derived JSONL files.
        """

    @abstractmethod
    def write(
        self, destination: Path, split: DatasetSplit, texts: Iterable[str]
    ) -> int:
        """Replace one split's derived JSONL with the given stories.

        Each story is one JSON object on its own line, including when the
        story text itself contains line breaks. A failed write leaves any
        previous derived file for that split in place.

        Args:
            destination: Directory that holds derived JSONL files.
            split: Split being written.
            texts: Story text in source order.

        Returns:
            The number of stories written.
        """
