"""Plain text value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Text:
    """Characters of one text body.

    Two texts are the same value when ``body`` matches exactly. ``body`` is a
    string, including the empty string.
    """

    body: str

    def __post_init__(self) -> None:
        """Reject a body that is not a string.

        Args:
            None.

        Returns:
            None.
        """
        if not isinstance(self.body, str):
            message = "Text body must be a string"
            raise TypeError(message)
