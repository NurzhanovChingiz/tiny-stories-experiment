"""Pre-compiled cleaning regex patterns with match timeouts."""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

REGEX_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class CompiledCleanPatterns:
    """Page-header and junk-line patterns compiled for one run.

    Attributes:
        page_header: Pattern that matches a page-header line.
        junk_line: Pattern that matches a junk line.
    """

    page_header: re.Pattern[str]
    junk_line: re.Pattern[str]

    @classmethod
    def compile_patterns(
        cls,
        page_header_pattern: str,
        junk_line_pattern: str,
    ) -> CompiledCleanPatterns:
        """Compile the two caller-supplied line patterns.

        Args:
            page_header_pattern: Regular expression for page-header lines.
            junk_line_pattern: Regular expression for junk lines.

        Returns:
            Both patterns compiled with the multiline flag.
        """
        return cls(
            page_header=re.compile(page_header_pattern, re.MULTILINE),
            junk_line=re.compile(junk_line_pattern, re.MULTILINE),
        )
