"""Turn one story into whitespace token ids."""

from __future__ import annotations

from tiny_story.application.ports.text_tokenizer import TextTokenizer
from tiny_story.domain.entities.tokenization import TokenizeTextResult
from tiny_story.infrastructure.text_processing.tokenize_text.steps import encode_tokens


class WhitespaceTokenizer(TextTokenizer):
    """Map story text with one fixed vocabulary."""

    def __init__(self, *, enabled: bool, vocabulary: dict[str, int]) -> None:
        """Store whether tokenization runs and which ids it uses.

        Args:
            enabled: Whether text is encoded. When false, the id list is empty.
            vocabulary: Token to id, including the special tokens.

        Returns:
            None.
        """
        self._enabled = enabled
        self._vocabulary = vocabulary

    def tokenize(self, content: str) -> tuple[tuple[int, ...], TokenizeTextResult]:
        """Encode one story, or return no ids when tokenization is off.

        Args:
            content: Story text.

        Returns:
            The ids and how many were produced.

        Flow:
            1. Return an empty encoding when tokenization is off.
            2. Encode the story with the stored vocabulary.
        """
        # 1. Return an empty encoding when tokenization is off.
        if not self._enabled:
            return (), TokenizeTextResult(token_count=0)

        # 2. Encode the story with the stored vocabulary.
        ids = encode_tokens(content, self._vocabulary)
        return ids, TokenizeTextResult(token_count=len(ids))
