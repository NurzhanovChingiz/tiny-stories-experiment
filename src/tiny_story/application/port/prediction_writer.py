"""PredictionWriter gateway — writes predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod

type JsonValue = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
)


class PredictionWriter(ABC):
    """Abstract base for prediction writer."""

    @abstractmethod
    def write(self, name: str, prediction: dict[str, JsonValue]) -> None:
        """Write a prediction.

        Args:
            name: The name of the prediction.
            prediction: The prediction to write.

        Returns:
            None.
        """
