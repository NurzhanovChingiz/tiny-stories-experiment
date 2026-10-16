"""Read one UTF-8 .jsonl file into text values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tiny_story.application.ports.text_source import TextSource
from tiny_story.domain.entities.text import Text

if TYPE_CHECKING:
    from pathlib import Path


class ReadTextJsonl(TextSource):
    """Read story text from a local ``.jsonl`` file."""

    JSONL_SUFFIX = ".jsonl"
    TEXT_FIELD = "text"

    def read(self, path: Path) -> tuple[Text, ...]:
        """Read one local ``.jsonl`` file into text values.

        Each non-empty line is one JSON object. The string ``text`` field
        becomes one text value, in file order. A blank line is skipped. Any
        other suffix is refused.

        Args:
            path: File to read.

        Returns:
            One text for each JSON object, in file order.

        Raises:
            ValueError: The suffix is not ``.jsonl``.
            TypeError: A line is not an object, or it has no string ``text`` field.
            json.JSONDecodeError: A non-empty line is not JSON.

        Flow:
            1. Refuse a suffix that is not ``.jsonl``.
            2. Return one text for each JSON object.
        """
        # 1. Refuse a suffix that is not .jsonl.
        suffix = path.suffix.lower()
        if suffix != self.JSONL_SUFFIX:
            message = f"Unsupported file extension: {suffix}"
            raise ValueError(message)

        # 2. Return one text for each JSON object.
        texts: list[Text] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() == "":
                    continue
                texts.append(Text(self._text_body(line)))
        return tuple(texts)

    def _text_body(self, line: str) -> str:
        """Return the string ``text`` field of one JSONL line.

        Args:
            line: One non-empty line from the file.

        Returns:
            The story text stored on that line.

        Raises:
            TypeError: The line is not an object, or it has no string ``text`` field.
            json.JSONDecodeError: The line is not JSON.
        """
        parsed: object = json.loads(line)
        if not isinstance(parsed, dict):
            message = "JSONL line must be an object"
            raise TypeError(message)
        body: object = parsed.get(self.TEXT_FIELD)
        if not isinstance(body, str):
            message = "JSONL object must contain a string text field"
            raise TypeError(message)
        return body
