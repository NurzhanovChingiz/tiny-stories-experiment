"""Split story text and assign stable whitespace token ids."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

PAD = "<pad>"
UNKNOWN = "<unk>"
START = "<bos>"
END = "<eos>"
SPECIAL_TOKENS = (PAD, UNKNOWN, START, END)


def whitespace_tokens(content: str) -> tuple[str, ...]:
    """Split story text on whitespace.

    Args:
        content: Story text. Repeated whitespace collapses into one split.

    Returns:
        The tokens in story order. Empty when the text has no tokens.
    """
    return tuple(content.split())


def build_vocabulary(tokens: Iterable[str]) -> dict[str, int]:
    """Assign ids to specials and then to the sorted unique story tokens.

    Args:
        tokens: Whitespace tokens collected from the stories.

    Returns:
        Token to id. ``<pad>`` is 0, ``<unk>`` is 1, ``<bos>`` is 2, and
        ``<eos>`` is 3. Story tokens follow in alphabetical order.

    Flow:
        1. Reserve the special token ids.
        2. Assign the remaining ids in alphabetical order.
    """
    # 1. Reserve the special token ids.
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}

    # 2. Assign the remaining ids in alphabetical order.
    next_id = len(SPECIAL_TOKENS)
    for token in sorted(set(tokens)):
        if token in vocabulary:
            continue
        vocabulary[token] = next_id
        next_id += 1
    return vocabulary


def encode_tokens(content: str, vocabulary: dict[str, int]) -> tuple[int, ...]:
    """Map one story to ids, wrapped with start and end markers.

    A token missing from ``vocabulary`` uses the unknown-token id.

    Args:
        content: Story text.
        vocabulary: Token to id, including the special tokens.

    Returns:
        ``<bos>``, one id per whitespace token, then ``<eos>``.

    Flow:
        1. Start the sequence, map each whitespace token, and end it.
    """
    # 1. Start the sequence, map each whitespace token, and end it.
    unknown = vocabulary[UNKNOWN]
    ids = [
        vocabulary[START],
        *(vocabulary.get(token, unknown) for token in whitespace_tokens(content)),
        vocabulary[END],
    ]
    return tuple(ids)
