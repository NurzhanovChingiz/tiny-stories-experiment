"""Tiny tokenized corpus used by Lightning tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

VOCABULARY = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "hello": 4,
    "world": 5,
}


def write_tokenized_corpus(directory: Path, stories: int = 8) -> None:
    """Write a tiny vocabulary and JSONL stories with token ids.

    Args:
        directory: Destination directory.
        stories: Number of identical short stories to write.

    Returns:
        None.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(
        json.dumps(VOCABULARY),
        encoding="utf-8",
    )
    token_ids = [2, 4, 5, 4, 5, 3]
    lines = [
        json.dumps({"text": "hello world hello world", "token_ids": token_ids})
        for _ in range(stories)
    ]
    (directory / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
