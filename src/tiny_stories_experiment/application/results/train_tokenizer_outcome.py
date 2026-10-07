"""Result of training a tokenizer on prepared split files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TrainTokenizerOutcome:
    """Tokenizer document written from the prepared splits.

    Attributes:
        processed_directory: Directory whose split files were read.
        artifact_path: Stored ``tokenizer.json``.
        vocab_size: Number of tokens in the trained vocabulary.
        story_count: Number of stories read from the prepared splits.
    """

    processed_directory: Path
    artifact_path: Path
    vocab_size: int
    story_count: int
