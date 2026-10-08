"""Causal LM spec constraints."""

from __future__ import annotations

import pytest

from tiny_stories_experiment.domain.errors import InvalidModelSpecError
from tiny_stories_experiment.domain.modeling.model_spec import ModelSpec


def test_model_spec_accepts_divisible_positive_sizes() -> None:
    """A width that splits evenly across heads can be built."""
    spec = ModelSpec(
        vocab_size=32,
        embedding_size=16,
        layer_count=2,
        attention_head_count=4,
        context_length=8,
        feedforward_size=32,
    )

    assert spec.vocab_size == 32
    assert spec.embedding_size == 16
    assert spec.layer_count == 2
    assert spec.attention_head_count == 4
    assert spec.context_length == 8
    assert spec.feedforward_size == 32


def test_tiny_debug_spec_exposes_overfit_sizes() -> None:
    """The named debug spec is a legal tiny causal LM."""
    spec = ModelSpec.tiny_debug()

    assert spec.vocab_size == 64
    assert spec.embedding_size == 64
    assert spec.layer_count == 2
    assert spec.attention_head_count == 4
    assert spec.context_length == 32
    assert spec.feedforward_size == 128
    assert spec.embedding_size % spec.attention_head_count == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"vocab_size": 1}, "vocab size"),
        ({"vocab_size": 0}, "vocab size"),
        ({"embedding_size": 0}, "embedding size"),
        ({"embedding_size": -4}, "embedding size"),
        ({"layer_count": 0}, "layer count"),
        ({"attention_head_count": 0}, "attention head count"),
        ({"attention_head_count": 3}, "must divide"),
        ({"context_length": 1}, "context length"),
        ({"context_length": 0}, "context length"),
        ({"feedforward_size": 0}, "feedforward size"),
        ({"feedforward_size": -8}, "feedforward size"),
    ],
)
def test_model_spec_rejects_impossible_sizes(
    overrides: dict[str, int],
    match: str,
) -> None:
    """Heads, widths, depth, context, and vocab must be buildable."""
    values = {
        "vocab_size": 32,
        "embedding_size": 16,
        "layer_count": 1,
        "attention_head_count": 4,
        "context_length": 8,
        "feedforward_size": 32,
    }
    values.update(overrides)

    with pytest.raises(InvalidModelSpecError, match=match):
        ModelSpec(**values)
