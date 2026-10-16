"""Shared story text for acceptance scenarios."""

from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, then

from tests.acceptance.story_text import story_body

_PAGE_NUMBER_LINE_WITH_TRAILING_SPACE = "Page | 2 "
_THREE_SPACES = "   "


@pytest.fixture
def cleaning_switch() -> str:
    """Return the shipped cleaning switch.

    Returns:
        ``on`` when the scenario does not turn cleaning off.
    """
    return "on"


@pytest.fixture
def sentence_marking_switch() -> str:
    """Return the shipped sentence-marking switch.

    Returns:
        ``on`` when the scenario does not turn sentence marking off.
    """
    return "on"


@given("a raw story:", target_fixture="story_text")
def raw_story(docstring: str) -> str:
    """Store a story that may span several lines.

    Args:
        docstring: Story text from the scenario.

    Returns:
        The story text.
    """
    return story_body(docstring)


@given(parsers.parse('a story that reads "{story}"'), target_fixture="story_text")
def story_that_reads(story: str) -> str:
    """Store a single-line story.

    Args:
        story: Story text from the scenario.

    Returns:
        The story text.
    """
    return story


@given("a page-number line that ends with a space", target_fixture="story_text")
def page_number_line_with_trailing_space() -> str:
    """Store a page-number line whose only extra character is a space.

    Returns:
        The page-number line.
    """
    return _PAGE_NUMBER_LINE_WITH_TRAILING_SPACE


@given("an empty story", target_fixture="story_text")
def empty_story() -> str:
    """Store a story with no characters.

    Returns:
        An empty string.
    """
    return ""


@given("a story of three spaces", target_fixture="story_text")
def story_of_three_spaces() -> str:
    """Store a story that contains only spaces.

    Returns:
        Three spaces.
    """
    return _THREE_SPACES


@then(parsers.parse('the prepared story reads "{text}"'))
def prepared_story_reads(prepared_story: str, text: str) -> None:
    """Assert the prepared story text.

    Args:
        prepared_story: Text produced by the scenario.
        text: Text the scenario expects.
    """
    assert prepared_story == text


@then("the prepared story is:")
def prepared_story_is(prepared_story: str, docstring: str) -> None:
    """Assert a prepared story that may span several lines.

    Args:
        prepared_story: Text produced by the scenario.
        docstring: Expected story text.
    """
    assert prepared_story == story_body(docstring)


@then("the prepared story is empty")
def prepared_story_is_empty(prepared_story: str) -> None:
    """Assert that preparation removed every character.

    Args:
        prepared_story: Text produced by the scenario.
    """
    assert prepared_story == ""


@then("the story text is unchanged")
def story_text_is_unchanged(story_text: str, prepared_story: str) -> None:
    """Assert that preparation returned the original story.

    Args:
        story_text: Story text before preparation.
        prepared_story: Text produced by the scenario.
    """
    assert prepared_story == story_text
