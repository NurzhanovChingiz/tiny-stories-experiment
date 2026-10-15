"""RunLogger gateway — logs run messages."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RunLogger(ABC):
    """Abstract base for run logger."""

    @abstractmethod
    def log(self, message: str, *, level: str = "info") -> None:
        """Log a message.

        Args:
            message: Message to log.
            level: Level of the message.

        Returns:
            None.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the run logger."""
