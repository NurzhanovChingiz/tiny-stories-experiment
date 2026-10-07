"""Training contract for one BPE tokenizer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tiny_stories_experiment.domain.errors import InvalidTokenizerSpecError

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class TokenizerSpec:
    """Limits and identity for one BPE tokenizer training run.

    Attributes:
        name: Single path segment used as the artifact directory name.
        vocab_size: Maximum number of tokens, including special tokens.
        min_frequency: Smallest pair count that may become a merge.
        special_tokens: Tokens reserved ahead of learned merges, in id order.
    """

    name: str
    vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        """Refuse a name, size, frequency, or special-token list that cannot be stored.

        Flow:
            1. Artifact name — require one portable path segment.
            2. Vocab size — require room beyond the reserved special tokens.
            3. Minimum frequency — require a positive pair count.
            4. Special tokens — require unique, non-empty token strings.
        """
        # 1. Artifact name
        if _NAME.fullmatch(self.name) is None:
            message = (
                f"Tokenizer name {self.name!r} is not a single portable path segment."
            )
            raise InvalidTokenizerSpecError(message)

        # 2. Vocab size
        if self.vocab_size <= len(self.special_tokens):
            message = (
                f"Tokenizer vocab size {self.vocab_size} must be greater than "
                f"{len(self.special_tokens)} special tokens."
            )
            raise InvalidTokenizerSpecError(message)

        # 3. Minimum frequency
        if self.min_frequency < 1:
            message = (
                f"Tokenizer min frequency {self.min_frequency} must be at least 1."
            )
            raise InvalidTokenizerSpecError(message)

        # 4. Special tokens
        if any(not token for token in self.special_tokens) or len(
            set(self.special_tokens)
        ) != len(self.special_tokens):
            message = "Tokenizer special tokens must be unique and non-empty."
            raise InvalidTokenizerSpecError(message)
