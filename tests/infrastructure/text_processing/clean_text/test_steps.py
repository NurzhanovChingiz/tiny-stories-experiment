"""Unit tests for cleaning steps."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from tiny_story.domain.entities.cleaning import CleanStep
from tiny_story.infrastructure.text_processing.clean_text import steps as clean_steps
from tiny_story.infrastructure.text_processing.clean_text.steps import (
    collapse_whitespace,
    filter_characters,
    join_hyphenated_line_breaks,
    reflow_hard_wrapped_lines,
    remove_page_headers,
)

if TYPE_CHECKING:
    import pytest

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


def test_join_hyphenated_line_breaks_counts_joins_during_the_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hyphen breaks are joined and counted in one pattern scan."""
    counter = _ScanCounter(clean_steps.__dict__["_HYPHEN_BREAK"])
    monkeypatch.setattr(clean_steps, "_HYPHEN_BREAK", counter)

    cleaned, step = join_hyphenated_line_breaks(
        "for You-\nKnow-Who has gone", enabled=True
    )

    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 1)
    assert (cleaned, step) == (
        "for You-Know-Who has gone",
        CleanStep(
            name="join_hyphenated_line_breaks",
            applied=True,
            detail="joined 1 hyphen breaks",
        ),
    )

    unchanged, idle = join_hyphenated_line_breaks("no break here", enabled=True)

    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 2)
    assert (unchanged, idle) == (
        "no break here",
        CleanStep(name="join_hyphenated_line_breaks", applied=False, detail=None),
    )


class _ScanCounter:
    """Counts pattern scans delegated to one compiled expression."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        """Store the pattern whose scans this counter records.

        Args:
            pattern: Compiled expression that performs the real match.

        Returns:
            None.
        """
        self._pattern = pattern
        self.sub_calls = 0
        self.findall_calls = 0
        self.subn_calls = 0

    def sub(self, replacement: str, text: str, count: int = 0) -> str:
        """Record a separate replacement scan.

        Args:
            replacement: Text written in place of each match.
            text: Text being scanned.
            count: Maximum substitutions. Zero means no maximum.

        Returns:
            The replaced text.
        """
        self.sub_calls += 1
        return self._pattern.sub(replacement, text, count)

    def findall(self, text: str, pos: int = 0, endpos: int = sys.maxsize) -> list[str]:
        """Record a separate match-list scan.

        Args:
            text: Text being scanned.
            pos: Start index.
            endpos: End index.

        Returns:
            The matched pieces.
        """
        self.findall_calls += 1
        return self._pattern.findall(text, pos, endpos)

    def subn(self, replacement: str, text: str, count: int = 0) -> tuple[str, int]:
        """Record one replacement that also returns its count.

        Args:
            replacement: Text written in place of each match.
            text: Text being scanned.
            count: Maximum substitutions. Zero means no maximum.

        Returns:
            The replaced text and the number of substitutions.
        """
        self.subn_calls += 1
        return self._pattern.subn(replacement, text, count)


def test_filter_characters_counts_replacements_during_the_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disallowed characters are replaced and counted in one pattern scan."""
    counter = _ScanCounter(clean_steps.__dict__["_ALLOWED_CHARS"])
    monkeypatch.setattr(clean_steps, "_ALLOWED_CHARS", counter)

    cleaned, step = filter_characters("Hello, world!")

    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 1)
    assert (cleaned, step) == (
        "Hello  world!",
        CleanStep(
            name="filter_characters",
            applied=True,
            detail="filtered 1 chars",
        ),
    )

    unchanged, idle = filter_characters("Hello")

    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 2)
    assert (unchanged, idle) == (
        "Hello",
        CleanStep(name="filter_characters", applied=False, detail=None),
    )


def test_collapse_whitespace_collapses_runs_without_a_regex_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace runs collapse in one split, not a regular-expression scan."""
    counter = _ScanCounter(re.compile(r"\s+"))
    monkeypatch.setattr(clean_steps, "_WHITESPACE", counter, raising=False)

    cleaned, step = collapse_whitespace("  hello   world \n ")
    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 0)
    assert cleaned == "hello world"
    assert step == CleanStep(
        name="collapse_whitespace",
        applied=True,
        detail=None,
    )

    unchanged, idle = collapse_whitespace("hello world")
    assert (counter.sub_calls, counter.findall_calls, counter.subn_calls) == (0, 0, 0)
    assert unchanged == "hello world"
    assert idle == CleanStep(
        name="collapse_whitespace",
        applied=False,
        detail=None,
    )


def test_reflow_hard_wrapped_lines_uppercase_continuation() -> None:
    """A wrapped line that continues with a capital is still joined."""
    text = "they were the last people youd \nExpect to be involved"
    cleaned, step = reflow_hard_wrapped_lines(text, enabled=True)
    assert "youd Expect" in cleaned
    assert step.applied is True


def test_reflow_joins_several_wrapped_lines_and_keeps_a_stopped_line() -> None:
    """Several wrapped lines become one line, and a non-letter ending stays split."""
    cleaned, step = reflow_hard_wrapped_lines("alpha \nbeta \ngamma", enabled=True)
    assert cleaned == "alpha beta gamma"
    assert step == CleanStep(
        name="reflow_hard_wrapped_lines",
        applied=True,
        detail="joined 2 wrapped lines",
    )

    stopped, idle = reflow_hard_wrapped_lines("alpha.\nbeta", enabled=True)
    assert stopped == "alpha.\nbeta"
    assert idle == CleanStep(name="reflow_hard_wrapped_lines", applied=False)


def test_reflow_does_not_rescan_the_accumulated_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapped line is joined without searching the accumulated line again."""
    real = clean_steps.__dict__["_WRAP_END"]

    def _forbid_search(*_args: object, **_kwargs: object) -> None:
        message = "reflow rescanned the accumulated line"
        raise AssertionError(message)

    monkeypatch.setattr(
        clean_steps,
        "_WRAP_END",
        type("_Ending", (), {"match": real.match, "search": _forbid_search})(),
    )
    cleaned, step = reflow_hard_wrapped_lines(
        "they were the last people youd \nexpect to be involved",
        enabled=True,
    )
    assert cleaned == "they were the last people youd expect to be involved"
    assert step.applied is True
