"""Unit tests for cleaning steps."""

from __future__ import annotations

from tiny_story.infrastructure.text_processing.clean_text.steps import (
    join_hyphenated_line_breaks,
    reflow_hard_wrapped_lines,
    remove_page_headers,
)

_PAGE_HEADER = r"^Page \| \d+ .*$"


def test_remove_page_headers() -> None:
    """A page-header line is removed and counted."""
    text = "hello\nPage | 2 Harry Potter\nworld"
    cleaned, step = remove_page_headers(text, _PAGE_HEADER)
    assert "Page |" not in cleaned
    assert step.applied is True
    assert step.detail == "removed 1 headers"


def test_reflow_hard_wrapped_lines() -> None:
    """A line broken after a word is joined with the next line."""
    text = "they were the last people youd \nexpect to be involved"
    cleaned, step = reflow_hard_wrapped_lines(text, enabled=True)
    assert "youd expect" in cleaned
    assert step.applied is True


def test_join_hyphenated_line_breaks() -> None:
    """A hyphen at a line break stays attached to the continued word."""
    text = "for You-\nKnow-Who has gone"
    cleaned, step = join_hyphenated_line_breaks(text, enabled=True)
    assert "You-Know-Who" in cleaned
    assert step.applied is True


def test_reflow_hard_wrapped_lines_uppercase_continuation() -> None:
    """A wrapped line that continues with a capital is still joined."""
    text = "they were the last people youd \nExpect to be involved"
    cleaned, step = reflow_hard_wrapped_lines(text, enabled=True)
    assert "youd Expect" in cleaned
    assert step.applied is True
