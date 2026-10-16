"""Normalization step functions."""

from __future__ import annotations

import re

from tiny_story.domain.entities.normalization import NormalizeStep

_SENT_PUNCT = ".!?"
_SPECIALS = '[".!?'
_SPECIAL_PATTERN = re.compile(rf"(?<!\s)([{re.escape(_SPECIALS)}])")
_AFTER_OPEN = re.compile(r'(["\[])(?=\w)')
_WHITESPACE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> tuple[str, NormalizeStep]:
    """Collapse whitespace runs and strip ends.

    Args:
        text: Text that may contain repeated whitespace.

    Returns:
        The collapsed text and the step outcome.
    """
    cleaned = _WHITESPACE.sub(" ", text).strip()
    return cleaned, NormalizeStep(
        name="collapse_whitespace",
        applied=cleaned != text,
        detail=None,
    )


def fix_attached_punctuation(text: str) -> tuple[str, NormalizeStep]:
    """Detach . ! ? from the start, end, or middle of alphanumeric tokens.

    Args:
        text: Text that may attach sentence punctuation to words.

    Returns:
        The spaced text and the step outcome.
    """
    cleaned = text
    cleaned = re.sub(rf"(\w)([{_SENT_PUNCT}])", r"\1 \2", cleaned)
    cleaned = re.sub(rf"([{_SENT_PUNCT}])(\w)", r"\1 \2", cleaned)
    cleaned = re.sub(rf"(\w)([{_SENT_PUNCT}])(\w)", r"\1 \2 \3", cleaned)
    return cleaned, NormalizeStep(
        name="fix_attached_punctuation",
        applied=cleaned != text,
        detail=None,
    )


def space_before_specials(text: str) -> tuple[str, NormalizeStep]:
    """Ensure a space before special token characters.

    Args:
        text: Text that may attach quotes, brackets, or sentence punctuation.

    Returns:
        The spaced text and the step outcome.
    """
    cleaned = _SPECIAL_PATTERN.sub(r" \1", text)
    cleaned = _AFTER_OPEN.sub(r"\1 ", cleaned)
    return cleaned, NormalizeStep(
        name="space_before_specials",
        applied=cleaned != text,
        detail=None,
    )


def append_sentence_delimiters(
    text: str,
    sentence_delimiter: str,
) -> tuple[str, NormalizeStep]:
    """Insert the sentence delimiter after . ! ? tokens.

    Args:
        text: Text whose sentence punctuation should be marked.
        sentence_delimiter: One-character delimiter inserted after . ! ?.

    Returns:
        The marked text and the step outcome.

    Flow:
        1. Insert the delimiter after sentence punctuation in the middle.
        2. Insert the delimiter after sentence punctuation at the end.
    """
    # 1. Insert the delimiter after sentence punctuation in the middle.
    mid_pattern = re.compile(r" ([.!?])(?!\s*\|)")
    eos_pattern = re.compile(r"([.!?])(?!\s*\|)$")
    cleaned, mid_count = mid_pattern.subn(rf" \1{sentence_delimiter} ", text)

    # 2. Insert the delimiter after sentence punctuation at the end.
    cleaned, eos_count = eos_pattern.subn(rf"\1{sentence_delimiter}", cleaned)
    match_count = mid_count + eos_count
    detail = f"inserted {match_count} delimiters" if match_count else None
    return cleaned, NormalizeStep(
        name="append_sentence_delimiters",
        applied=bool(match_count),
        detail=detail,
    )
