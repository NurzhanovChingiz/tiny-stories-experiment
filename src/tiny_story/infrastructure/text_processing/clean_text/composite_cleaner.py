"""Composite text cleaner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_story.application.ports.text_cleaner import TextCleaner
from tiny_story.domain.entities.cleaning import CleanTextResult
from tiny_story.domain.errors import CleanTextError
from tiny_story.infrastructure.text_processing.clean_text.compiled_patterns import (
    CompiledCleanPatterns,
)
from tiny_story.infrastructure.text_processing.clean_text.steps import (
    collapse_whitespace,
    filter_characters,
    flatten_newlines,
    join_hyphenated_line_breaks,
    normalize_quotes,
    reflow_hard_wrapped_lines,
    remove_junk_lines,
    remove_page_headers,
)

if TYPE_CHECKING:
    from tiny_story.domain.entities.cleaning import CleanStep


class CompositeTextCleaner(TextCleaner):
    """Runs the text-cleaning steps in a fixed order."""

    def __init__(
        self,
        *,
        enabled: bool,
        page_header_pattern: str,
        junk_line_pattern: str,
        reflow_hard_wrapped_lines: bool,
        join_hyphenated_line_breaks: bool,
    ) -> None:
        """Store the flags and compile the line patterns once.

        Args:
            enabled: Whether cleaning runs. When false, text is unchanged.
            page_header_pattern: Regular expression for page-header lines.
            junk_line_pattern: Regular expression for junk lines.
            reflow_hard_wrapped_lines: Whether to join lines broken mid-sentence.
            join_hyphenated_line_breaks: Whether to join hyphenated line breaks.

        Returns:
            None.

        Flow:
            1. Store the flags for this cleaner.
            2. Compile the two line patterns once when cleaning is on.
        """
        # 1. Store the flags for this cleaner.
        self._enabled = enabled
        self._page_header_pattern = page_header_pattern
        self._junk_line_pattern = junk_line_pattern
        self._reflow_hard_wrapped_lines = reflow_hard_wrapped_lines
        self._join_hyphenated_line_breaks = join_hyphenated_line_breaks

        # 2. Compile the two line patterns once when cleaning is on.
        self._patterns = (
            CompiledCleanPatterns.compile_patterns(
                page_header_pattern,
                junk_line_pattern,
            )
            if enabled
            else None
        )

    def clean(self, content: str) -> tuple[str, CleanTextResult]:
        """Clean text through the fixed step order.

        Args:
            content: Raw text content.

        Returns:
            The cleaned text and the step outcomes. Steps are empty when
            cleaning is disabled.

        Raises:
            CleanTextError: Cleaning is on and the line patterns were not compiled.

        Flow:
            1. Return the original text when cleaning is disabled.
            2. Run each cleaning step in order and keep its outcome.
            3. Return the cleaned text and the recorded outcomes.
        """
        # 1. Return the original text when cleaning is disabled.
        if not self._enabled:
            return content, CleanTextResult(
                original_length=len(content),
                cleaned_length=len(content),
                steps=(),
            )

        # 2. Run each cleaning step in order and keep its outcome.
        original_length = len(content)
        patterns = self._patterns
        if patterns is None:
            message = "Cleaning patterns were not compiled"
            raise CleanTextError(message)
        text, steps = _apply_steps(
            content,
            patterns,
            page_header_pattern=self._page_header_pattern,
            junk_line_pattern=self._junk_line_pattern,
            reflow_wrapped_lines=self._reflow_hard_wrapped_lines,
            join_hyphen_breaks=self._join_hyphenated_line_breaks,
        )

        # 3. Return the cleaned text and the recorded outcomes.
        return text, CleanTextResult(
            original_length=original_length,
            cleaned_length=len(text),
            steps=tuple(steps),
        )


def _apply_steps(
    text: str,
    patterns: CompiledCleanPatterns,
    *,
    page_header_pattern: str,
    junk_line_pattern: str,
    reflow_wrapped_lines: bool,
    join_hyphen_breaks: bool,
) -> tuple[str, list[CleanStep]]:
    """Apply the fixed cleaning sequence.

    Args:
        text: Text entering the sequence.
        patterns: Compiled page-header and junk-line patterns.
        page_header_pattern: Regular expression for page-header lines.
        junk_line_pattern: Regular expression for junk lines.
        reflow_wrapped_lines: Whether to join lines broken mid-sentence.
        join_hyphen_breaks: Whether to join hyphenated line breaks.

    Returns:
        The text after every cleaning step and the outcomes in pipeline order.
    """
    steps: list[CleanStep] = []
    text, step = remove_page_headers(text, page_header_pattern, patterns)
    steps.append(step)
    text, step = remove_junk_lines(text, junk_line_pattern, patterns)
    steps.append(step)
    text, step = reflow_hard_wrapped_lines(text, enabled=reflow_wrapped_lines)
    steps.append(step)
    text, step = join_hyphenated_line_breaks(text, enabled=join_hyphen_breaks)
    steps.append(step)
    text, step = flatten_newlines(text)
    steps.append(step)
    text, step = collapse_whitespace(text)
    steps.append(step)
    text, step = normalize_quotes(text)
    steps.append(step)
    text, step = filter_characters(text)
    steps.append(step)
    text, step = collapse_whitespace(text)
    steps.append(step)
    return text, steps
