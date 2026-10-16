"""Normalize the text field of a JSONL story file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tiny_story.domain.errors import NormalizeTextError
from tiny_story.infrastructure.text_processing.normalize_text.composite_normalizer import (
    CompositeTextNormalizer,
)


class JsonlStoryNormalizer:
    """Rewrite each story ``text`` field into a new JSONL file."""

    def __init__(self, *, enabled: bool, sentence_delimiter: str) -> None:
        """Store the normalization flags used for every story.

        Args:
            enabled: Whether normalization runs. When false, text is unchanged.
            sentence_delimiter: One-character marker inserted after . ! ?.

        Returns:
            None.
        """
        self._normalizer = CompositeTextNormalizer(
            enabled=enabled,
            sentence_delimiter=sentence_delimiter,
        )

    def normalize_file(self, source: Path, destination: Path) -> None:
        """Write a derived JSONL file with each story text normalized.

        The source file is only read. A failed rewrite removes its temporary
        file and leaves an existing destination in place.

        Args:
            source: Raw JSONL file. Each object has a string ``text`` field.
            destination: Derived JSONL file. It must not be the source path.

        Returns:
            None.

        Raises:
            NormalizeTextError: The source is missing, the destination would
                replace the source, a line is not a story object, or the derived
                file could not be written.

        Flow:
            1. Reject a missing source or a destination that is the source.
            2. Rewrite the stories into a temporary file.
            3. Publish the temporary file as the destination.
        """
        # 1. Reject a missing source or a destination that is the source.
        if not source.is_file():
            message = f"Story file does not exist: {source}"
            raise NormalizeTextError(message)
        if source.resolve() == destination.resolve():
            message = f"Refusing to replace story file {source}"
            raise NormalizeTextError(message)

        # 2. Rewrite the stories into a temporary file.
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(destination)
        try:
            _rewrite_stories(source, temporary, self._normalizer)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not normalize {source.name}"
            raise NormalizeTextError(message) from exc
        except NormalizeTextError:
            temporary.unlink(missing_ok=True)
            raise

        # 3. Publish the temporary file as the destination.
        try:
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not normalize {source.name}"
            raise NormalizeTextError(message) from exc


def _temporary_path(destination: Path) -> Path:
    """Return an exclusive temporary path beside the destination.

    Args:
        destination: Final derived file.

    Returns:
        The temporary path.

    Raises:
        OSError: The temporary file could not be created.
    """
    descriptor, name = tempfile.mkstemp(
        prefix=".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _rewrite_stories(
    source: Path,
    temporary: Path,
    normalizer: CompositeTextNormalizer,
) -> None:
    """Copy stories to the temporary file with normalized text.

    Args:
        source: Raw JSONL file.
        temporary: Exclusive file that receives the derived lines.
        normalizer: Normalizer used for each story text.

    Returns:
        None.

    Raises:
        NormalizeTextError: A line is not a JSON object with a string ``text``.
        json.JSONDecodeError: A non-empty line is not JSON.
        OSError: A file could not be read or written.
        TypeError: JSON parsing returned a type the line walker does not accept.
    """
    with (
        source.open(encoding="utf-8") as incoming,
        temporary.open(
            "w",
            encoding="utf-8",
        ) as outgoing,
    ):
        for line_number, line in enumerate(incoming, start=1):
            if line.strip() == "":
                continue
            outgoing.write(_normalized_line(line, line_number, normalizer))


def _normalized_line(
    line: str,
    line_number: int,
    normalizer: CompositeTextNormalizer,
) -> str:
    """Return one JSONL line whose ``text`` field has been normalized.

    Args:
        line: One non-empty JSONL line.
        line_number: 1-based line number used in errors.
        normalizer: Normalizer used for the story text.

    Returns:
        The derived JSON object plus a newline.

    Raises:
        NormalizeTextError: The line is not an object, or ``text`` is not a string.
        json.JSONDecodeError: The line is not JSON.
    """
    parsed: object = json.loads(line)
    if not isinstance(parsed, dict):
        message = f"JSONL line {line_number} must be an object"
        raise NormalizeTextError(message)
    body = parsed.get("text")
    if not isinstance(body, str):
        message = f"JSONL line {line_number} must contain a string text field"
        raise NormalizeTextError(message)
    normalized, _result = normalizer.normalize(body)
    parsed["text"] = normalized
    return json.dumps(parsed, ensure_ascii=False) + "\n"
