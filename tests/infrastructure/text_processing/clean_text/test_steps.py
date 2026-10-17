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
    assert cleaned == "hello\nworld"
    assert step.applied is True
    assert step.detail == "removed 1 headers"


def test_remove_page_headers_keeps_text_after_a_non_lf_break() -> None:
    """A header does not consume story text after a non-LF break."""
    for separator in ("\r", "\f", "\u2028"):
        cleaned, step = remove_page_headers(
            f"Page | 2 Title{separator}The end.",
            _PAGE_HEADER,
        )
        assert cleaned == "The end."
        assert step.applied is True
        assert step.detail == "removed 1 headers"


def test_remove_page_headers_removes_a_header_after_a_carriage_return() -> None:
    """A header that starts after a carriage return is removed."""
    cleaned, step = remove_page_headers(
        "Intro\rPage | 2 Title\rThe end.",
        _PAGE_HEADER,
    )
    assert cleaned == "Intro\nThe end."
    assert step.applied is True


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
