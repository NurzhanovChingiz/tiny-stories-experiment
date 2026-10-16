"""Unit tests for normalization steps."""

from __future__ import annotations

from tiny_story.infrastructure.text_processing.clean_text.composite_cleaner import (
    CompositeTextCleaner,
)
from tiny_story.infrastructure.text_processing.normalize_text.composite_normalizer import (
    CompositeTextNormalizer,
)
from tiny_story.infrastructure.text_processing.normalize_text.steps import (
    append_sentence_delimiters,
)


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
