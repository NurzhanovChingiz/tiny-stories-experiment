"""Text cleaning outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleanStep:
    """Outcome of one cleaning step.

    Attributes:
        name: Step function name.
        applied: Whether the step changed the text or recorded a match.
        detail: Short count of what the step changed, when there is one.
    """

    name: str
    applied: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CleanTextResult:
    """Lengths and step outcomes for one cleaning run.

    Attributes:
        original_length: Character count before cleaning.
        cleaned_length: Character count after cleaning.
        steps: Step outcomes in pipeline order. Empty when cleaning is off.
    """

    original_length: int
    cleaned_length: int
    steps: tuple[CleanStep, ...]
