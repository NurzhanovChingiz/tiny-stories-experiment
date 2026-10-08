"""Token-id batches a causal model spec can score."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_stories_experiment.domain.errors import InvalidTokenBatchError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tiny_stories_experiment.domain.modeling.model_spec import ModelSpec

_NEXT_TOKEN_PAIR_LENGTH = 2


def validated_token_rows(
    spec: ModelSpec,
    token_ids: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return token rows that ``spec`` can score as a next-token batch.

    Args:
        spec: Vocabulary size and context length that bound the batch.
        token_ids: Candidate rows of token ids.

    Returns:
        A rectangular tuple of integer rows.

    Raises:
        InvalidTokenBatchError: The rows are empty, shorter than a next-token
            pair, longer than the context, ragged, or outside the vocabulary.

    Flow:
        1. Row shape — require one rectangle that fits the context.
        2. Vocabulary — require every id to be an integer inside the vocab.
    """
    # 1. Row shape
    _require_scoreable_rectangle(spec, token_ids)
    # 2. Vocabulary
    return tuple(_vocabulary_row(spec.vocab_size, row) for row in token_ids)


def _require_scoreable_rectangle(
    spec: ModelSpec,
    token_ids: Sequence[Sequence[int]],
) -> None:
    """Refuse a batch that is not one next-token rectangle inside the context.

    Args:
        spec: Context length that bounds each row.
        token_ids: Candidate rows of token ids.

    Raises:
        InvalidTokenBatchError: The rows are empty, shorter than a next-token
            pair, longer than the context, or ragged.
    """
    if len(token_ids) == 0:
        message = "Token batch must contain at least one row."
        raise InvalidTokenBatchError(message)
    width = len(token_ids[0])
    if width < _NEXT_TOKEN_PAIR_LENGTH:
        message = "Token batch rows must contain at least two ids."
        raise InvalidTokenBatchError(message)
    if width > spec.context_length:
        message = (
            f"Token batch row length {width} exceeds context length "
            f"{spec.context_length}."
        )
        raise InvalidTokenBatchError(message)
    for row in token_ids:
        if len(row) != width:
            message = "Token batch rows must be the same length."
            raise InvalidTokenBatchError(message)


def _vocabulary_row(vocab_size: int, row: Sequence[int]) -> tuple[int, ...]:
    """Copy one row after refusing an id the vocabulary cannot score.

    Args:
        vocab_size: Exclusive upper bound for a legal id.
        row: One candidate row of token ids.

    Returns:
        The same ids as a tuple.

    Raises:
        InvalidTokenBatchError: An id is a boolean or lies outside the vocabulary.
    """
    checked: list[int] = []
    for token_id in row:
        if _id_outside_vocabulary(token_id, vocab_size):
            message = (
                f"Token id {token_id!r} is outside the vocabulary of size {vocab_size}."
            )
            raise InvalidTokenBatchError(message)
        checked.append(token_id)
    return tuple(checked)


def _id_outside_vocabulary(token_id: object, vocab_size: int) -> bool:
    """Return whether ``token_id`` is not an in-range vocabulary index.

    Args:
        token_id: Candidate id. Booleans are refused even though ``bool`` is an int.
        vocab_size: Exclusive upper bound for a legal id.

    Returns:
        ``True`` when the id cannot be scored.
    """
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        return True
    return token_id < 0 or token_id >= vocab_size
