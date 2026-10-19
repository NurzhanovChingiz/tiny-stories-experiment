"""Tests for the composite cleaner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_story.infrastructure.text_processing.clean_text.compiled_patterns import (
    CompiledCleanPatterns,
)
from tiny_story.infrastructure.text_processing.clean_text.composite_cleaner import (
    CompositeTextCleaner,
)

if TYPE_CHECKING:
    import pytest

_PAGE_HEADER = r"^Page \| \d+ .*$"
_JUNK_LINE = r"^[/\\-]+$"


def _count_compiles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count ``compile_patterns`` calls and return a one-element tally.

    Args:
        monkeypatch: Pytest patcher for the compile classmethod.

    Returns:
        A list whose first item is the number of compiles observed.
    """
    tally = [0]
    real_compile = CompiledCleanPatterns.compile_patterns

    def _count(
        page_header_pattern: str,
        junk_line_pattern: str,
    ) -> CompiledCleanPatterns:
        tally[0] += 1
        return real_compile(page_header_pattern, junk_line_pattern)

    monkeypatch.setattr(CompiledCleanPatterns, "compile_patterns", _count)
    return tally


def test_clean_compiles_line_patterns_once_per_cleaner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many stories reuse the patterns compiled when the cleaner is built."""
    tally = _count_compiles(monkeypatch)
    cleaner = CompositeTextCleaner(
        enabled=True,
        page_header_pattern=_PAGE_HEADER,
        junk_line_pattern=_JUNK_LINE,
        reflow_hard_wrapped_lines=True,
        join_hyphenated_line_breaks=True,
    )
    story = "Page | 2 Title\n----\nThe cat sat."
    cleaned, result = cleaner.clean(story)
    cleaner.clean(story)
    cleaner.clean(story)

    assert tally[0] == 1
    assert cleaned == "The cat sat."
    assert result.cleaned_length == len(cleaned)


def test_disabled_cleaner_does_not_compile_line_patterns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleaner that is off does not compile patterns it will not use."""
    tally = _count_compiles(monkeypatch)
    cleaner = CompositeTextCleaner(
        enabled=False,
        page_header_pattern=_PAGE_HEADER,
        junk_line_pattern=_JUNK_LINE,
        reflow_hard_wrapped_lines=True,
        join_hyphenated_line_breaks=True,
    )

    cleaned, result = cleaner.clean("Page | 2 Title")

    assert tally[0] == 0
    assert cleaned == "Page | 2 Title"
    assert result.steps == ()
