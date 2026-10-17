"""Normalization step functions."""

from __future__ import annotations

import re

from tiny_story.domain.entities.normalization import NormalizeStep

_SENT_PUNCT = ".!?"
_SPECIALS = '[".!?'
_WORD_BEFORE_PUNCT = re.compile(rf"(\w)([{_SENT_PUNCT}])")
_PUNCT_BEFORE_WORD = re.compile(rf"([{_SENT_PUNCT}])(\w)")
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
    cleaned = _WORD_BEFORE_PUNCT.sub(r"\1 \2", text)
    cleaned = _PUNCT_BEFORE_WORD.sub(r"\1 \2", cleaned)
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


def compile_sentence_mark_patterns(
    sentence_delimiter: str,
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile the middle and end sentence-mark patterns for one delimiter.

    Args:
        sentence_delimiter: One-character delimiter inserted after . ! ?.

    Returns:
        The middle pattern and the end-of-string pattern.
    """
    literal_delimiter = re.escape(sentence_delimiter)
    return (
        re.compile(rf" ([.!?])(?!\s*{literal_delimiter})"),
        re.compile(rf"([.!?])(?!\s*{literal_delimiter})$"),
    )


def append_sentence_delimiters(
    text: str,
    sentence_delimiter: str,
    *,
    patterns: tuple[re.Pattern[str], re.Pattern[str]] | None = None,
) -> tuple[str, NormalizeStep]:
    """Insert the sentence delimiter after . ! ? tokens.

    Args:
        text: Text whose sentence punctuation should be marked.
        sentence_delimiter: One-character delimiter inserted after . ! ?.
        patterns: Middle and end patterns already compiled for this delimiter.

    Returns:
        The marked text and the step outcome.

    Flow:
        1. Insert the literal delimiter after unmarked sentence punctuation in the middle.
        2. Insert the literal delimiter after an unmarked sentence mark attached to the final word.
    """
    mid_pattern, eos_pattern = patterns or compile_sentence_mark_patterns(
        sentence_delimiter
    )

    def _after_middle(match: re.Match[str]) -> str:
        return f" {match.group(1)}{sentence_delimiter} "

    def _after_end(match: re.Match[str]) -> str:
        return f"{match.group(1)}{sentence_delimiter}"

    # 1. Insert the literal delimiter after unmarked sentence punctuation in the middle.
    cleaned, mid_count = mid_pattern.subn(_after_middle, text)

    # 2. Insert the literal delimiter after an unmarked sentence mark attached to the final word.
    final_mark_is_spaced = (
        len(text) >= 2 and text[-2] == " " and text[-1] in _SENT_PUNCT
    )
    eos_count = 0
    if text[-1:] in _SENT_PUNCT and not final_mark_is_spaced:
        cleaned, eos_count = eos_pattern.subn(_after_end, cleaned)
    match_count = mid_count + eos_count
    detail = f"inserted {match_count} delimiters" if match_count else None
    return cleaned, NormalizeStep(
        name="append_sentence_delimiters",
        applied=bool(match_count),
        detail=detail,
    )
