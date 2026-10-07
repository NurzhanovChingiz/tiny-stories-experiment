"""Serialized tokenizer produced by one training run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainedTokenizer:
    """Serialized tokenizer produced by one training run.

    Attributes:
        content: UTF-8 JSON document for ``tokenizer.json``.
        vocab_size: Number of tokens in the trained vocabulary.
        story_count: Number of stories the trainer read.
    """

    content: bytes
    vocab_size: int
    story_count: int
