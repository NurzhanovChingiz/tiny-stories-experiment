"""Text normalization outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizeStep:
    """Outcome of one normalization step.

    Attributes:
        name: Step function name.
        applied: Whether the step changed the text or recorded a match.
        detail: Short count of what the step changed, when there is one.
    """

    name: str
    applied: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizeTextResult:
    """Lengths and step outcomes for one normalization run.

    Attributes:
        original_length: Character count before normalization.
        normalized_length: Character count after normalization.
        steps: Step outcomes in pipeline order. Empty when normalization is off.
    """

    original_length: int
    normalized_length: int
    steps: tuple[NormalizeStep, ...]
