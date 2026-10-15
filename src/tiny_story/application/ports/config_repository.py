"""Configuration abstract base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T")


class ConfigRepository(ABC):
    """Port for accessing application configurations."""

    @abstractmethod
    def get(self, config_type: type[T]) -> T:
        """Retrieve a typed configuration.

        Args:
            config_type: The type of configuration to retrieve.

        Returns:
            The configuration.
        """
        ...
