"""What to do when a raw dataset file is already stored."""

from __future__ import annotations

from enum import StrEnum


class ExistingFilePolicy(StrEnum):
    """Whether a stored raw file is kept or acquired again."""

    KEEP = "keep"
    REPLACE = "replace"
