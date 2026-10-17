"""Whitespace tokenization outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizeTextResult:
    """How many ids one story text became.

    Attributes:
        token_count: Ids written for the story, including the start and end
            markers. Zero when tokenization is off.
    """

    token_count: int
