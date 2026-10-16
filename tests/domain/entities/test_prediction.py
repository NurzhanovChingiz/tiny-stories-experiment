"""Behavior of the model prediction value."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from tiny_story.domain.entities.prediction import Prediction


def test_prediction_keeps_the_label_and_score() -> None:
    """The stored result is the label and score the model produced."""
    prediction = Prediction(label="ok", score=1)

    assert prediction.label == "ok"
    assert prediction.score == 1


@pytest.mark.parametrize(
    ("label", "score"),
    [("ok", 1), ("", 0), ("ok", -1)],
)
def test_predictions_match_when_the_label_and_score_match(
    label: str, score: int
) -> None:
    """Equality follows the exact label and score."""
    assert Prediction(label=label, score=score) == Prediction(label=label, score=score)


def test_predictions_differ_when_the_label_or_score_differs() -> None:
    """A different label or score is a different prediction."""
    original = Prediction(label="ok", score=1)

    assert original != Prediction(label="fail", score=1)
    assert original != Prediction(label="ok", score=0)


def test_equal_predictions_share_a_hash() -> None:
    """Equal predictions can share a set or dictionary key."""
    left = Prediction(label="ok", score=1)
    right = Prediction(label="ok", score=1)

    assert hash(left) == hash(right)
    assert {left, right} == {Prediction(label="ok", score=1)}


@pytest.mark.parametrize(
    ("label", "score"),
    [
        (cast("str", 1), 1),
        ("ok", cast("int", 1.5)),
    ],
)
def test_prediction_rejects_unexpected_types(label: str, score: int) -> None:
    """A label that is not text, or a score that is not an integer, is rejected."""
    with pytest.raises(
        TypeError,
        match="Prediction label must be a string and score must be an integer",
    ):
        Prediction(label=label, score=score)


def test_prediction_rejects_a_boolean_score() -> None:
    """A boolean is not an integer score."""
    score = True

    with pytest.raises(
        TypeError,
        match="Prediction label must be a string and score must be an integer",
    ):
        Prediction(label="ok", score=cast("int", score))


def test_prediction_fields_cannot_be_replaced() -> None:
    """The label and score stay the values given at construction."""
    prediction = Prediction(label="ok", score=1)

    for field_name in ("label", "score"):
        with pytest.raises(FrozenInstanceError):
            setattr(prediction, field_name, "other")
