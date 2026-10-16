"""Composite text normalizer."""

from __future__ import annotations

from tiny_story.application.ports.text_normalizer import TextNormalizer
from tiny_story.domain.entities.normalization import NormalizeStep, NormalizeTextResult
from tiny_story.infrastructure.text_processing.normalize_text.steps import (
    append_sentence_delimiters,
    collapse_whitespace,
    fix_attached_punctuation,
    space_before_specials,
)


class CompositeTextNormalizer(TextNormalizer):
    """Runs the sentence-formatting steps in a fixed order."""

    def __init__(self, *, enabled: bool, sentence_delimiter: str) -> None:
        """Store the flags for one normalizer.

        Args:
            enabled: Whether normalization runs. When false, text is unchanged.
            sentence_delimiter: One-character marker inserted after . ! ?.

        Returns:
            None.
        """
        self._enabled = enabled
        self._sentence_delimiter = sentence_delimiter

    def normalize(self, content: str) -> tuple[str, NormalizeTextResult]:
        """Normalize text through the fixed step order.

        Cleaning is a separate stage. This stage spaces punctuation and inserts
        the sentence delimiter.

        Args:
            content: Text to format, usually already cleaned.

        Returns:
            The normalized text and the step outcomes. Steps are empty when
            normalization is disabled.

        Flow:
            1. Return the original text when normalization is disabled.
            2. Run each formatting step in order and keep its outcome.
            3. Return the normalized text and the recorded outcomes.
        """
        # 1. Return the original text when normalization is disabled.
        if not self._enabled:
            return content, NormalizeTextResult(
                original_length=len(content),
                normalized_length=len(content),
                steps=(),
            )

        # 2. Run each formatting step in order and keep its outcome.
        original_length = len(content)
        text, steps = _apply_steps(content, sentence_delimiter=self._sentence_delimiter)

        # 3. Return the normalized text and the recorded outcomes.
        return text, NormalizeTextResult(
            original_length=original_length,
            normalized_length=len(text),
            steps=tuple(steps),
        )


def _apply_steps(
    text: str,
    *,
    sentence_delimiter: str,
) -> tuple[str, list[NormalizeStep]]:
    """Apply the fixed normalization sequence.

    Args:
        text: Text entering the sequence.
        sentence_delimiter: One-character marker inserted after . ! ?.

    Returns:
        The text after every formatting step and the outcomes in pipeline order.
    """
    steps: list[NormalizeStep] = []
    text, step = fix_attached_punctuation(text)
    steps.append(step)
    text, step = collapse_whitespace(text)
    steps.append(step)
    text, step = space_before_specials(text)
    steps.append(step)
    text, step = append_sentence_delimiters(text, sentence_delimiter)
    steps.append(step)
    text, step = collapse_whitespace(text)
    steps.append(step)
    return text, steps
