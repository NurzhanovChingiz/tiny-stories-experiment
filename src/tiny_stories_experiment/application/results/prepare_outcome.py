"""Result of writing derived story text for each raw split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit


@dataclass(frozen=True)
class PreparedSplit:
    """One split written into the processed directory.

    Attributes:
        raw_filename: Raw JSONL file the stories were read from.
        split: Published split named by that file.
        story_count: Number of stories written.
        output_name: Derived JSONL file name inside the processed directory.
    """

    raw_filename: str
    split: DatasetSplit
    story_count: int
    output_name: str


@dataclass(frozen=True)
class PrepareOutcome:
    """Derived split files produced from a raw dataset.

    Attributes:
        raw_directory: Directory that held the untouched raw files.
        processed_directory: Directory that received the derived JSONL files.
        splits: One entry per raw file, in spec order.
    """

    raw_directory: Path
    processed_directory: Path
    splits: tuple[PreparedSplit, ...]
