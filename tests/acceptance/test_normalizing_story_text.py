"""Acceptance scenarios for marking sentences in a story."""

from __future__ import annotations

from pytest_bdd import given, scenarios, when

from tiny_story.infrastructure.text_processing.normalize_text.composite_normalizer import (
    CompositeTextNormalizer,
)

scenarios("features/normalizing_story_text.feature")

_SENTENCE_MARK = "|"


def normalize_text(story_text: str, sentence_marking_switch: str) -> str:
    """Mark sentences with the shipped delimiter.

    Args:
        story_text: Story text, usually already cleaned.
        sentence_marking_switch: ``on`` when sentence marking runs.

    Returns:
        The normalized story text.
    """
    normalizer = CompositeTextNormalizer(
        enabled=sentence_marking_switch == "on",
        sentence_delimiter=_SENTENCE_MARK,
    )
    normalized, _result = normalizer.normalize(story_text)
    return normalized


@given("sentence marking is turned off", target_fixture="sentence_marking_switch")
def sentence_marking_is_turned_off() -> str:
    """Turn sentence marking off for this scenario.

    Returns:
        ``off``.
    """
    return "off"


@when(
    "the story is normalized with the shipped sentence mark",
    target_fixture="prepared_story",
)
def the_story_is_normalized(story_text: str, sentence_marking_switch: str) -> str:
    """Normalize the scenario story.

    Args:
        story_text: Story text before sentence marking.
        sentence_marking_switch: ``on`` when sentence marking runs.

    Returns:
        The normalized story text.
    """
    return normalize_text(story_text, sentence_marking_switch)
