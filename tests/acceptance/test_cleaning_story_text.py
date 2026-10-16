"""Acceptance scenarios for cleaning a story."""

from __future__ import annotations

from pytest_bdd import given, scenarios, when

from tiny_story.infrastructure.text_processing.clean_text.composite_cleaner import (
    CompositeTextCleaner,
)

scenarios("features/cleaning_story_text.feature")

_PAGE_HEADER_PATTERN = r"^Page \| \d+ .*$"
_JUNK_LINE_PATTERN = r"^[/\\-]+$"


def clean_text(story_text: str, cleaning_switch: str) -> str:
    """Clean one story with the shipped character and header rules.

    Args:
        story_text: Raw story text.
        cleaning_switch: ``on`` when cleaning runs.

    Returns:
        The cleaned story text.
    """
    cleaner = CompositeTextCleaner(
        enabled=cleaning_switch == "on",
        page_header_pattern=_PAGE_HEADER_PATTERN,
        junk_line_pattern=_JUNK_LINE_PATTERN,
        reflow_hard_wrapped_lines=True,
        join_hyphenated_line_breaks=True,
    )
    cleaned, _result = cleaner.clean(story_text)
    return cleaned


@given("story cleaning is turned off", target_fixture="cleaning_switch")
def story_cleaning_is_turned_off() -> str:
    """Turn story cleaning off for this scenario.

    Returns:
        ``off``.
    """
    return "off"


@when("the story is cleaned", target_fixture="prepared_story")
def the_story_is_cleaned(story_text: str, cleaning_switch: str) -> str:
    """Clean the scenario story.

    Args:
        story_text: Raw story text.
        cleaning_switch: ``on`` when cleaning runs.

    Returns:
        The cleaned story text.
    """
    return clean_text(story_text, cleaning_switch)
