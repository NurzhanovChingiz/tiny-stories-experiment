"""Clean the text field of a JSONL story file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tiny_story.domain.errors import CleanTextError
from tiny_story.infrastructure.text_processing.clean_text.composite_cleaner import (
    CompositeTextCleaner,
)


class JsonlStoryCleaner:
    """Rewrite each story ``text`` field into a cleaned JSONL file."""

    def __init__(
        self,
        *,
        enabled: bool,
        page_header_pattern: str,
        junk_line_pattern: str,
        reflow_hard_wrapped_lines: bool,
        join_hyphenated_line_breaks: bool,
    ) -> None:
        """Store the cleaning flags used for every story.

        Args:
            enabled: Whether cleaning runs. When false, text is unchanged.
            page_header_pattern: Regular expression for page-header lines.
            junk_line_pattern: Regular expression for junk lines.
            reflow_hard_wrapped_lines: Whether to join lines broken mid-sentence.
            join_hyphenated_line_breaks: Whether to join hyphenated line breaks.

        Returns:
            None.
        """
        self._cleaner = CompositeTextCleaner(
            enabled=enabled,
            page_header_pattern=page_header_pattern,
            junk_line_pattern=junk_line_pattern,
            reflow_hard_wrapped_lines=reflow_hard_wrapped_lines,
            join_hyphenated_line_breaks=join_hyphenated_line_breaks,
        )

    def clean_file(self, source: Path, destination: Path) -> None:
        """Write a derived JSONL file with each story text cleaned.

        The source file is only read. A failed rewrite removes its temporary
        file and leaves an existing destination in place.

        Args:
            source: Raw JSONL file. Each object has a string ``text`` field.
            destination: Derived JSONL file. It must not be the source path.

        Returns:
            None.

        Raises:
            CleanTextError: The source is missing, the destination would replace
                the source, a line is not a story object, or the derived file
                could not be written.

        Flow:
            1. Reject a missing source or a destination that is the source.
            2. Rewrite the stories into a temporary file.
            3. Publish the temporary file as the destination.
        """
        # 1. Reject a missing source or a destination that is the source.
        if not source.is_file():
            message = f"Raw story file does not exist: {source}"
            raise CleanTextError(message)
        if source.resolve() == destination.resolve():
            message = f"Refusing to replace raw story file {source}"
            raise CleanTextError(message)

        # 2. Rewrite the stories into a temporary file.
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(destination)
        try:
            _rewrite_stories(source, temporary, self._cleaner)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not clean {source.name}"
            raise CleanTextError(message) from exc
        except CleanTextError:
            temporary.unlink(missing_ok=True)
            raise

        # 3. Publish the temporary file as the destination.
        try:
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            message = f"Could not clean {source.name}"
            raise CleanTextError(message) from exc


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
    cleaner: CompositeTextCleaner,
) -> None:
    """Copy stories to the temporary file with cleaned text.

    Args:
        source: Raw JSONL file.
        temporary: Exclusive file that receives the derived lines.
        cleaner: Cleaner used for each story text.

    Returns:
        None.

    Raises:
        CleanTextError: A line is not a JSON object with a string ``text``.
        json.JSONDecodeError: A non-empty line is not JSON.
        OSError: A file could not be read or written.
        TypeError: JSON parsing returned a type the line walker does not accept.
    """
    with (
        source.open(encoding="utf-8") as incoming,
        temporary.open("w", encoding="utf-8") as outgoing,
    ):
        for line_number, line in enumerate(incoming, start=1):
            if line.strip() == "":
                continue
            outgoing.write(_cleaned_line(line, line_number, cleaner))


def _cleaned_line(line: str, line_number: int, cleaner: CompositeTextCleaner) -> str:
    """Return one JSONL line whose ``text`` field has been cleaned.

    Args:
        line: One non-empty JSONL line.
        line_number: 1-based line number used in errors.
        cleaner: Cleaner used for the story text.

    Returns:
        The derived JSON object plus a newline.

    Raises:
        CleanTextError: The line is not an object, or ``text`` is not a string.
        json.JSONDecodeError: The line is not JSON.
    """
    parsed: object = json.loads(line)
    if not isinstance(parsed, dict):
        message = f"JSONL line {line_number} must be an object"
        raise CleanTextError(message)
    body = parsed.get("text")
    if not isinstance(body, str):
        message = f"JSONL line {line_number} must contain a string text field"
        raise CleanTextError(message)
    cleaned, _result = cleaner.clean(body)
    parsed["text"] = cleaned
    return json.dumps(parsed, ensure_ascii=False) + "\n"
