"""JSONL adapter that yields the text field from a raw story file."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tiny_stories_experiment.application.ports.raw_story_source import RawStorySource
from tiny_stories_experiment.domain.errors import RawStoryRecordError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class JsonlStorySource(RawStorySource):
    """Raw stories stored as one JSON object per line."""

    def texts(self, path: Path) -> Iterator[str]:
        """Yield each story's text from a raw JSONL file.

        Args:
            path: Raw JSONL file to read.

        Returns:
            Story text in file order.

        Raises:
            RawStoryRecordError: A line is not an object with a string text field.

        Flow:
            1. Read lines — open the raw file without writing it.
            2. Story field — yield the string text of each JSON object.
        """
        # 1. Read lines
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                # 2. Story field
                yield _story_text(line, path, line_number)


def _story_text(line: str, path: Path, line_number: int) -> str:
    """Return the story text from one raw JSONL line.

    Args:
        line: One raw file line.
        path: Raw file the line came from.
        line_number: One-based line number inside that file.

    Returns:
        The string ``text`` field.

    Raises:
        RawStoryRecordError: The line is not an object with a string text field.

    Flow:
        1. JSON object — parse the line, or refuse invalid JSON.
        2. Text field — return the string text, or refuse another shape.
    """
    # 1. JSON object
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        message = f"Raw file {path} line {line_number} is not a story record."
        raise RawStoryRecordError(message) from error
    # 2. Text field
    text = record.get("text") if isinstance(record, dict) else None
    if not isinstance(text, str):
        message = f"Raw file {path} line {line_number} is not a story record."
        raise RawStoryRecordError(message)
    return text
