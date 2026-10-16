"""Behavior of the text value."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from tiny_story.domain.entities.text import Text


def test_text_keeps_the_body() -> None:
    """The stored value is the body that was given."""
    text = Text(body="Once upon a time.")

    assert text.body == "Once upon a time."


@pytest.mark.parametrize("body", ["Once upon a time.", "", "café"])
def test_texts_match_when_the_body_matches(body: str) -> None:
    """Equality follows the exact body."""
    assert Text(body=body) == Text(body=body)


def test_texts_differ_when_the_body_differs() -> None:
    """A different body is a different text."""
    assert Text(body="one") != Text(body="two")


def test_equal_texts_share_a_hash() -> None:
    """Equal texts can share a set or dictionary key."""
    left = Text(body="one")
    right = Text(body="one")

    assert hash(left) == hash(right)
    assert {left, right} == {Text(body="one")}


def test_text_rejects_a_non_string_body() -> None:
    """A body that is not text is rejected."""
    with pytest.raises(TypeError, match="Text body must be a string"):
        Text(body=cast("str", 1))


def test_text_body_cannot_be_replaced() -> None:
    """The body stays the value given at construction."""
    text = Text(body="one")

    for field_name in ("body",):
        with pytest.raises(FrozenInstanceError):
            setattr(text, field_name, "other")
