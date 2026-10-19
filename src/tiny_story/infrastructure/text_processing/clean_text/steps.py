"""Cleaning step functions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import regex as rx

from tiny_story.domain.entities.cleaning import CleanStep
from tiny_story.infrastructure.text_processing.clean_text.compiled_patterns import (
    REGEX_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from tiny_story.infrastructure.text_processing.clean_text.compiled_patterns import (
        CompiledCleanPatterns,
    )

_ALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9 \[\"\.!\?\|]")
_HYPHEN_BREAK = re.compile(r"-\s*\n\s*(\w)", re.MULTILINE)
_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "",
        "\u2019": "",
        "'": "",
    }
)
_UNICODE_LETTER = r"[^\W\d_]"
_WRAP_END = re.compile(rf"{_UNICODE_LETTER}\s*$", re.UNICODE)
_WRAP_START = re.compile(rf"^{_UNICODE_LETTER}", re.UNICODE)


def remove_page_headers(
    text: str,
    page_header_pattern: str,
    patterns: CompiledCleanPatterns | None = None,
) -> tuple[str, CleanStep]:
    """Remove lines matching the page header pattern.

    Args:
        text: Text that may contain page-header lines.
        page_header_pattern: Regular expression used when ``patterns`` is omitted.
        patterns: Patterns compiled for this run.

    Returns:
        The text without matching header lines, and the step outcome.

    Flow:
        1. Select the page-header pattern.
        2. Drop logical lines that match the pattern.
        3. Return the original text when no line matches.
    """
    # 1. Select the page-header pattern.
    pattern = patterns.page_header if patterns is not None else None
    if pattern is None:
        pattern = rx.compile(page_header_pattern, re.MULTILINE)

    # 2. Drop logical lines that match the pattern.
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if pattern.search(line, timeout=REGEX_TIMEOUT_SECONDS) is not None:
            removed += 1
            continue
        kept.append(line)
    if removed:
        detail = f"removed {removed} headers"
        return "\n".join(kept), CleanStep(
            name="remove_page_headers",
            applied=True,
            detail=detail,
        )

    # 3. Return the original text when no line matches.
    return text, CleanStep(
        name="remove_page_headers",
        applied=False,
        detail=None,
    )


def remove_junk_lines(
    text: str,
    junk_line_pattern: str,
    patterns: CompiledCleanPatterns | None = None,
) -> tuple[str, CleanStep]:
    """Drop lines that match the junk line pattern.

    Args:
        text: Text that may contain junk lines.
        junk_line_pattern: Regular expression used when ``patterns`` is omitted.
        patterns: Patterns compiled for this run.

    Returns:
        The kept lines joined by newlines, and the step outcome.

    Flow:
        1. Select the junk-line pattern.
        2. Drop matching lines and record how many were removed.
    """
    # 1. Select the junk-line pattern.
    pattern = patterns.junk_line if patterns is not None else None
    if pattern is None:
        pattern = rx.compile(junk_line_pattern, re.MULTILINE)

    # 2. Drop matching lines and record how many were removed.
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if not pattern.match(line.strip(), timeout=REGEX_TIMEOUT_SECONDS)
    ]
    removed = len(lines) - len(kept)
    cleaned = "\n".join(kept)
    detail = f"removed {removed} junk lines" if removed else None
    return cleaned, CleanStep(
        name="remove_junk_lines",
        applied=removed > 0,
        detail=detail,
    )


def _ends_with_unicode_letter(text: str) -> bool:
    """Return whether a letter is the last non-space character.

    Args:
        text: One logical line, without a line break.

    Returns:
        True when the wrap-end letter class matches the last non-space character.
    """
    index = len(text)
    while index > 0 and text[index - 1].isspace():
        index -= 1
    if index == 0:
        return False
    return _WRAP_END.match(text, index - 1, index) is not None


def reflow_hard_wrapped_lines(text: str, *, enabled: bool) -> tuple[str, CleanStep]:
    """Join lines broken mid-sentence.

    Args:
        text: Text that may contain hard-wrapped lines.
        enabled: Whether this step runs.

    Returns:
        The reflowed text and the step outcome.

    Flow:
        1. Leave the text unchanged when reflow is off or there are no lines.
        2. Join a line that continues a word by inspecting the previous ending.
        3. Record how many lines were joined.
    """
    # 1. Leave the text unchanged when reflow is off or there are no lines.
    if not enabled:
        return text, CleanStep(name="reflow_hard_wrapped_lines", applied=False)

    lines = text.splitlines()
    if not lines:
        return text, CleanStep(name="reflow_hard_wrapped_lines", applied=False)

    # 2. Join a line that continues a word by inspecting the previous ending.
    merged: list[str] = []
    pieces: list[str] = [lines[0]]
    joins = 0
    for line in lines[1:]:
        stripped = line.strip()
        if (
            stripped
            and _ends_with_unicode_letter(pieces[-1])
            and _WRAP_START.match(stripped)
        ):
            pieces[-1] = pieces[-1].rstrip()
            pieces.append(" ")
            pieces.append(stripped)
            joins += 1
        else:
            merged.append("".join(pieces))
            pieces = [line]

    merged.append("".join(pieces))
    cleaned = "\n".join(merged)

    # 3. Record how many lines were joined.
    detail = f"joined {joins} wrapped lines" if joins else None
    return cleaned, CleanStep(
        name="reflow_hard_wrapped_lines",
        applied=joins > 0,
        detail=detail,
    )


def join_hyphenated_line_breaks(text: str, *, enabled: bool) -> tuple[str, CleanStep]:
    """Join words split across lines with a hyphen.

    Args:
        text: Text that may split a hyphenated word across lines.
        enabled: Whether this step runs.

    Returns:
        The joined text and the step outcome.

    Flow:
        1. Leave the text unchanged when joining is off.
        2. Replace each hyphenated line break and count those replacements in the same scan.
    """
    # 1. Leave the text unchanged when joining is off.
    if not enabled:
        return text, CleanStep(name="join_hyphenated_line_breaks", applied=False)

    # 2. Replace each hyphenated line break and count those replacements in the same scan.
    cleaned, joined = _HYPHEN_BREAK.subn(r"-\1", text)
    detail = f"joined {joined} hyphen breaks" if joined else None
    return cleaned, CleanStep(
        name="join_hyphenated_line_breaks",
        applied=joined > 0,
        detail=detail,
    )


def flatten_newlines(text: str) -> tuple[str, CleanStep]:
    """Replace newline characters with spaces.

    Args:
        text: Text that may contain newlines.

    Returns:
        The single-line text and the step outcome.
    """
    had_newlines = "\n" in text or "\r" in text
    cleaned = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return cleaned, CleanStep(
        name="flatten_newlines",
        applied=had_newlines,
        detail=None,
    )


def collapse_whitespace(text: str) -> tuple[str, CleanStep]:
    """Collapse whitespace runs and strip ends.

    Args:
        text: Text that may contain repeated whitespace.

    Returns:
        The collapsed text and the step outcome.
    """
    cleaned = " ".join(text.split())
    return cleaned, CleanStep(
        name="collapse_whitespace",
        applied=cleaned != text,
        detail=None,
    )


def normalize_quotes(text: str) -> tuple[str, CleanStep]:
    """Normalize curly quotes to ASCII and strip apostrophes.

    Args:
        text: Text that may contain curly quotes or apostrophes.

    Returns:
        The translated text and the step outcome.
    """
    cleaned = text.translate(_TRANSLATION)
    return cleaned, CleanStep(
        name="normalize_quotes",
        applied=cleaned != text,
        detail=None,
    )


def filter_characters(text: str) -> tuple[str, CleanStep]:
    """Keep only allowed characters and map others to space.

    Args:
        text: Text that may contain disallowed characters.

    Returns:
        The filtered text and the step outcome.
    """
    cleaned, removed = _ALLOWED_CHARS.subn(" ", text)
    return cleaned, CleanStep(
        name="filter_characters",
        applied=cleaned != text,
        detail=f"filtered {removed} chars" if cleaned != text else None,
    )
