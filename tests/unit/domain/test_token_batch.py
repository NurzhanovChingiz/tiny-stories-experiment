"""Token batch constraints for a causal LM spec."""

from __future__ import annotations

import pytest

from tiny_stories_experiment.domain.errors import InvalidTokenBatchError
from tiny_stories_experiment.domain.modeling.model_spec import ModelSpec
from tiny_stories_experiment.domain.modeling.token_batch import validated_token_rows


def _spec() -> ModelSpec:
    return ModelSpec(
        vocab_size=8,
        embedding_size=4,
        layer_count=1,
        attention_head_count=2,
        context_length=4,
        feedforward_size=8,
    )


def test_validated_token_rows_keep_a_rectangular_batch() -> None:
    """In-vocab rows come back as tuples of the same length."""
    rows = validated_token_rows(_spec(), [[1, 2, 3], (3, 2, 1)])

    assert rows == ((1, 2, 3), (3, 2, 1))


@pytest.mark.parametrize(
    ("token_ids", "match"),
    [
        ((), "at least one row"),
        (((1,),), "at least two"),
        (((1, 2, 3, 4, 5),), "context length"),
        (((1, 2, 3), (1, 2)), "same length"),
        (((1, 8),), "outside the vocabulary"),
        (((1, -1),), "outside the vocabulary"),
        (((1, True),), "outside the vocabulary"),
    ],
)
def test_validated_token_rows_reject_unscoreable_batches(
    token_ids: tuple[tuple[int, ...], ...],
    match: str,
) -> None:
    """Empty, short, long, ragged, and out-of-vocab rows are refused."""
    with pytest.raises(InvalidTokenBatchError, match=match):
        validated_token_rows(_spec(), token_ids)
