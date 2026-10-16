"""Story text taken from Gherkin docstrings."""

from __future__ import annotations


def story_body(docstring: str) -> str:
    """Return story text from a Gherkin docstring.

    Args:
        docstring: Docstring attached to a story step.

    Returns:
        The story text without one trailing newline from the docstring.
    """
    if docstring.endswith("\n"):
        return docstring[:-1]
    return docstring
