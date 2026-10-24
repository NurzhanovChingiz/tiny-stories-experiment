"""Tests for seeded sequence splits."""

from __future__ import annotations

import pytest

from tiny_story.domain.errors import TrainError
from tiny_story.infrastructure.lightning.split import split_sequences

_SEQUENCES = tuple((index,) for index in range(10))


def test_split_sequences_is_stable_for_one_seed() -> None:
    """The same seed yields the same train slice."""
    first, _val, _test = split_sequences(
        _SEQUENCES,
        test_size=0.2,
        val_size=0.2,
        seed=7,
    )
    second, _val, _test = split_sequences(
        _SEQUENCES,
        test_size=0.2,
        val_size=0.2,
        seed=7,
    )

    assert first == second
    assert len(first) == 6


def test_split_sequences_keeps_disjoint_slices() -> None:
    """Train, validation, and test do not share sequences."""
    train, validation, test = split_sequences(
        _SEQUENCES,
        test_size=0.2,
        val_size=0.2,
        seed=7,
    )

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)


def test_split_sequences_rejects_an_empty_validation_split() -> None:
    """A collection too small for validation is rejected."""
    with pytest.raises(TrainError, match="validation"):
        split_sequences(((1, 2),), test_size=0.0, val_size=0.0, seed=0)
