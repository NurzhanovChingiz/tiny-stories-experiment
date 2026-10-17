"""Whitespace split and vocabulary ids."""

from __future__ import annotations

from tiny_story.infrastructure.text_processing.tokenize_text.steps import (
    END,
    PAD,
    START,
    UNKNOWN,
    build_vocabulary,
    encode_tokens,
    whitespace_tokens,
)
from tiny_story.infrastructure.text_processing.tokenize_text.whitespace_tokenizer import (
    WhitespaceTokenizer,
)


def test_whitespace_tokens_collapse_repeated_spaces() -> None:
    """A marked sentence splits on whitespace and keeps the delimiter token."""
    assert whitespace_tokens("hello world .|") == ("hello", "world", ".|")


def test_build_vocabulary_reserves_specials_then_sorts_words() -> None:
    """Specials occupy 0 through 3, and story tokens follow alphabetically."""
    vocabulary = build_vocabulary(["world", "hello", "hello", ".|"])

    assert vocabulary[PAD] == 0
    assert vocabulary[UNKNOWN] == 1
    assert vocabulary[START] == 2
    assert vocabulary[END] == 3
    assert vocabulary[".|"] == 4
    assert vocabulary["hello"] == 5
    assert vocabulary["world"] == 6


def test_encode_tokens_wraps_the_story_and_uses_unknown() -> None:
    """A known story is wrapped, and a missing token uses the unknown id."""
    vocabulary = build_vocabulary(["hello", "world"])

    assert encode_tokens("hello world", vocabulary) == (
        vocabulary[START],
        vocabulary["hello"],
        vocabulary["world"],
        vocabulary[END],
    )
    assert encode_tokens("hello there", vocabulary)[2] == vocabulary[UNKNOWN]


def test_whitespace_tokenizer_returns_no_ids_when_disabled() -> None:
    """Turning tokenization off leaves the id list empty."""
    tokenizer = WhitespaceTokenizer(
        enabled=False, vocabulary=build_vocabulary(["hello"])
    )

    ids, result = tokenizer.tokenize("hello")

    assert ids == ()
    assert result.token_count == 0
