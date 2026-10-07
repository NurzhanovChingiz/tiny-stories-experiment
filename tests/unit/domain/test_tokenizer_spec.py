"""Tokenizer spec constraints."""

from __future__ import annotations

import pytest

from tiny_stories_experiment.domain.errors import InvalidTokenizerSpecError
from tiny_stories_experiment.domain.tokenization.tokenizer_spec import TokenizerSpec


def test_tokenizer_spec_accepts_a_portable_name_and_limits() -> None:
    """A single path segment and a vocab above the special tokens is trainable."""
    spec = TokenizerSpec(
        name="tiny_stories_bpe",
        vocab_size=4096,
        min_frequency=2,
        special_tokens=("<|endoftext|>",),
    )

    assert spec.name == "tiny_stories_bpe"
    assert spec.vocab_size == 4096
    assert spec.special_tokens == ("<|endoftext|>",)


@pytest.mark.parametrize(
    ("name", "vocab_size", "min_frequency", "special_tokens"),
    [
        ("../tokenizer", 32, 1, ()),
        ("nested/name", 32, 1, ()),
        ("", 32, 1, ()),
        (".", 32, 1, ()),
        ("tiny", 1, 1, ("<|endoftext|>",)),
        ("tiny", 8, 0, ()),
        ("tiny", 8, 1, ("",)),
        ("tiny", 8, 1, ("<pad>", "<pad>")),
    ],
)
def test_tokenizer_spec_rejects_unstorable_limits(
    name: str,
    vocab_size: int,
    min_frequency: int,
    special_tokens: tuple[str, ...],
) -> None:
    """Names, sizes, frequencies, and special tokens must be storable."""
    with pytest.raises(InvalidTokenizerSpecError):
        TokenizerSpec(
            name=name,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
