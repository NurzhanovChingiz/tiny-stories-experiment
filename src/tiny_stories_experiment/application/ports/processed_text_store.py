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
    def stage(
        self, destination: Path, split: DatasetSplit, texts: Iterable[str]
    ) -> int:
        """Write one split's stories to a partial file.

        Each story is one JSON object on its own line, including when the
        story text itself contains line breaks. The current derived file stays
        in place. A failed stage removes its partial file.

        Args:
            destination: Directory that holds derived JSONL files.
            split: Split being staged.
            texts: Story text in source order.

        Returns:
            The number of stories written to the partial file.
        """

    @abstractmethod
    def publish(self, destination: Path, splits: tuple[DatasetSplit, ...]) -> None:
        """Replace derived files with staged partials, or restore the previous files.

        Args:
            destination: Directory that holds derived JSONL files.
            splits: Staged splits to publish, in spec order.
        """

    @abstractmethod
    def discard(self, destination: Path, splits: tuple[DatasetSplit, ...]) -> None:
        """Remove staged partials and leave derived files in place.

        Args:
            destination: Directory that holds derived JSONL files.
            splits: Staged splits whose partial files should be removed.
        """
