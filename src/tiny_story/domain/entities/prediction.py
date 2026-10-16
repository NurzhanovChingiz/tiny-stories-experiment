"""Model prediction value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    """Label and score the model produced.

    Two predictions are the same value when ``label`` and ``score`` match
    exactly. ``score`` is an integer, and a boolean is rejected.
    """

    label: str
    score: int

    def __post_init__(self) -> None:
        """Reject a label or score that has the wrong type.

        Args:
            None.

        Returns:
            None.
        """
        label_ok = isinstance(self.label, str)
        score_ok = isinstance(self.score, int) and not isinstance(self.score, bool)
        if not label_ok or not score_ok:
            message = "Prediction label must be a string and score must be an integer"
            raise TypeError(message)
