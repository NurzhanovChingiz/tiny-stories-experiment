"""Unit tests for normalization steps."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tiny_story.infrastructure.text_processing.clean_text.composite_cleaner import (
    CompositeTextCleaner,
)
from tiny_story.infrastructure.text_processing.normalize_text.composite_normalizer import (
    CompositeTextNormalizer,
)
from tiny_story.infrastructure.text_processing.normalize_text.steps import (
    append_sentence_delimiters,
    fix_attached_punctuation,
)

if TYPE_CHECKING:
    import pytest


def _normalizer(*, enabled: bool = True) -> CompositeTextNormalizer:
    """Return a normalizer that inserts the sentence delimiter.

    Args:
        enabled: Whether normalization runs.

    Returns:
        A composite normalizer.
    """
    return CompositeTextNormalizer(enabled=enabled, sentence_delimiter="|")


def _cleaner() -> CompositeTextCleaner:
    """Return a cleaner with the copied default patterns.

    Returns:
        A composite cleaner.
    """
    return CompositeTextCleaner(
        enabled=True,
        page_header_pattern=r"^Page \| \d+ .*$",
        junk_line_pattern=r"^[/\\-]+$",
        reflow_hard_wrapped_lines=True,
        join_hyphenated_line_breaks=True,
    )


def test_fix_attached_punctuation_spaces_words_around_a_period() -> None:
    """Attached words are spaced on both sides of a period."""
    cleaned, step = fix_attached_punctuation("Hello.World")
    assert cleaned == "Hello . World"
    assert step.applied is True


def test_delimiter_patterns_are_compiled_once_per_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stories reuse the delimiter patterns compiled for that normalizer."""
    compiles = 0
    real_compile = re.compile

    def _count_compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
        nonlocal compiles
        compiles += 1
        return real_compile(pattern, flags)

    monkeypatch.setattr(
        "tiny_story.infrastructure.text_processing.normalize_text.steps.re.compile",
        _count_compile,
    )
    normalizer = CompositeTextNormalizer(enabled=True, sentence_delimiter="|")
    compiled_for_normalizer = compiles

    normalizer.normalize("Stop.")
    normalizer.normalize("Hello.World")

    assert compiled_for_normalizer == 2
    assert compiles == compiled_for_normalizer


def test_append_sentence_delimiters() -> None:
    """Sentence punctuation in the middle receives a delimiter."""
    text = "hello world . They were fine"
    cleaned, step = append_sentence_delimiters(text, "|")
    assert ".| " in cleaned
    assert step.applied is True


def test_append_sentence_delimiters_end_of_string() -> None:
    """Sentence punctuation at the end receives a delimiter."""
    text = "hello world."
    cleaned, step = append_sentence_delimiters(text, "|")
    assert cleaned.endswith(".|")
    assert step.applied is True


def test_append_sentence_delimiters_spaced_eos_not_double_counted() -> None:
    """A spaced end-of-string mark is delimited once."""
    text = "hello world ."
    cleaned, step = append_sentence_delimiters(text, "|")
    assert cleaned == "hello world .| "
    assert step.detail == "inserted 1 delimiters"


def test_hash_delimiter_is_idempotent_on_attached_words() -> None:
    """A hash delimiter marks attached words once and stays stable."""
    normalizer = CompositeTextNormalizer(enabled=True, sentence_delimiter="#")
    once, _result = normalizer.normalize("Hello.World")
    twice, _again = normalizer.normalize(once)
    assert once == "Hello .# World"
    assert twice == "Hello .# World"


def test_hash_delimiter_marks_text_that_already_contains_a_pipe() -> None:
    """A pipe already in the text does not block a hash delimiter."""
    normalizer = CompositeTextNormalizer(enabled=True, sentence_delimiter="#")
    cleaned, _result = normalizer.normalize("Hello.|")
    assert cleaned == "Hello .# |"


def test_digit_and_backslash_delimiters_return_text() -> None:
    """A digit or backslash delimiter is literal text on marked and plain input."""
    for sentence_delimiter in ("1", "\\"):
        normalizer = CompositeTextNormalizer(
            enabled=True,
            sentence_delimiter=sentence_delimiter,
        )
        marked, _marked_result = normalizer.normalize("Stop.")
        plain, _plain_result = normalizer.normalize("hello")
        assert marked == f"Stop .{sentence_delimiter}"
        assert plain == "hello"


def test_pipe_delimiter_marks_stop() -> None:
    """The shipped pipe delimiter still spaces and marks a final period."""
    cleaned, _result = _normalizer().normalize("Stop.")
    assert cleaned == "Stop .|"


def test_normalizer_keeps_a_page_header_line() -> None:
    """Normalization formats punctuation and leaves header text in place."""
    cleaned, result = _normalizer().normalize("Page | 2 Harry Potter.")
    assert "Page" in cleaned
    assert cleaned.endswith(".|")
    assert any(step.applied for step in result.steps)


def test_composite_normalizer_disabled_passthrough() -> None:
    """A disabled normalizer returns the original text and no steps."""
    text = "hello\nworld"
    cleaned, result = _normalizer(enabled=False).normalize(text)
    assert cleaned == text
    assert result.steps == ()


def _book_snippet() -> str:
    """Return the hard-wrapped snippet used to separate the two stages.

    Returns:
        The snippet text.
    """
    return (
        "/\n\n"
        "cereal at the walls. \u201cLittle tyke,\u201d chortled Mr. Dursley\n"
        "as he left the house.\n"
        "Page | 3 Harry Potter and the Philosophers Stone - J.K. Rowling\n"
        "the cat."
    )


def test_cleaner_drops_headers_without_adding_a_delimiter() -> None:
    """Cleaning removes the page header and does not insert a sentence mark."""
    cleaned, clean_result = _cleaner().clean(_book_snippet())

    assert "\n" not in cleaned
    assert "Page |" not in cleaned
    assert "|" not in cleaned
    assert any(step.applied for step in clean_result.steps)


def test_normalizer_marks_sentences_in_cleaned_text() -> None:
    """Normalization inserts the delimiter into text the cleaner already produced."""
    cleaned, _clean_result = _cleaner().clean(_book_snippet())
    normalized, result = _normalizer().normalize(cleaned)

    assert "'" not in normalized
    assert ".| " in normalized
    assert '" Little tyke ' in normalized
    assert result.normalized_length == len(normalized)
