"""Load tokenized stories and expose padded token-id windows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from torch import long, tensor

from tiny_story.domain.errors import TrainError

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

_PAD_ID = 0


class TokenizedStoryDataset:
    """One padded token-id window per story sequence.

    Args:
        sequences: Equal-length integer windows, including start and end markers.
    """

    def __init__(self, sequences: tuple[tuple[int, ...], ...]) -> None:
        """Store the windows that this dataset iterates.

        Args:
            sequences: Equal-length integer windows.

        Returns:
            None.
        """
        self._sequences = sequences

    def __len__(self) -> int:
        """Return the number of windows.

        Returns:
            Window count.
        """
        return len(self._sequences)

    def __getitem__(self, index: int) -> Tensor:
        """Return one window as a long tensor.

        Args:
            index: Window index.

        Returns:
            Integer tensor of shape ``(time,)``.
        """
        return tensor(self._sequences[index], dtype=long)


def load_token_id_sequences(data_dir: Path) -> tuple[tuple[int, ...], ...]:
    """Read ``token_ids`` from every JSONL file in ``data_dir``.

    Purpose:
        Collect the tokenized stories that training will split and window.

    Strategy:
        Walk ``*.jsonl`` in name order, parse each non-blank line, and keep
        the integer ``token_ids`` list from each story object.

    Args:
        data_dir: Directory of tokenized JSONL files.

    Returns:
        Token-id sequences in file order, then line order.

    Raises:
        TrainError: The directory is missing, a line is not a story object
            with integer ``token_ids``, or no sequences were found.

    Flow:
        1. Reject a missing data directory.
        2. Collect token-id lists from each JSONL file.
        3. Reject an empty collection.
    """
    # 1. Reject a missing data directory.
    if not data_dir.is_dir():
        message = f"Tokenized data directory does not exist: {data_dir}"
        raise TrainError(message)

    # 2. Collect token-id lists from each JSONL file.
    sequences: list[tuple[int, ...]] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        sequences.extend(_sequences_from_file(path))

    # 3. Reject an empty collection.
    if not sequences:
        message = f"No tokenized stories found in {data_dir}"
        raise TrainError(message)
    return tuple(sequences)


def load_vocabulary_size(vocab_path: Path) -> int:
    """Return the number of tokens in a vocabulary JSON object.

    Args:
        vocab_path: JSON object mapping token strings to integer ids.

    Returns:
        Vocabulary size.

    Raises:
        TrainError: The file is missing, is not a JSON object, or is empty.
    """
    if not vocab_path.is_file():
        message = f"Vocabulary file does not exist: {vocab_path}"
        raise TrainError(message)
    try:
        parsed: object = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        message = f"Could not read vocabulary: {vocab_path}"
        raise TrainError(message) from exc
    if not isinstance(parsed, dict) or not parsed:
        message = f"Vocabulary must be a non-empty JSON object: {vocab_path}"
        raise TrainError(message)
    return len(parsed)


def window_sequences(
    sequences: tuple[tuple[int, ...], ...],
    sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    """Pad or truncate each sequence to ``sequence_length``.

    Args:
        sequences: Variable-length token-id lists.
        sequence_length: Target window length.

    Returns:
        One window per input sequence.

    Raises:
        TrainError: ``sequence_length`` is less than 2.
    """
    if sequence_length < 2:
        message = "sequence_length must be at least 2"
        raise TrainError(message)
    return tuple(_pad_or_truncate(ids, sequence_length) for ids in sequences)


def _sequences_from_file(path: Path) -> tuple[tuple[int, ...], ...]:
    """Read token-id lists from one JSONL file.

    Args:
        path: Tokenized story file.

    Returns:
        Sequences from non-blank lines.

    Raises:
        TrainError: A line cannot be decoded or lacks integer ``token_ids``.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        message = f"Could not read tokenized file: {path}"
        raise TrainError(message) from exc
    return tuple(
        _token_ids_from_line(line, line_number, path.name)
        for line_number, line in enumerate(lines, start=1)
        if line.strip() != ""
    )


def _token_ids_from_line(line: str, line_number: int, filename: str) -> tuple[int, ...]:
    """Parse one JSONL line into integer token ids.

    Args:
        line: One non-empty JSONL line.
        line_number: 1-based line number used in errors.
        filename: Source name used when the line is not JSON.

    Returns:
        The story's token ids.

    Raises:
        TrainError: The line is not an object with a non-empty integer list.
    """
    try:
        parsed: object = json.loads(line)
    except json.JSONDecodeError as exc:
        message = f"Could not read tokenized file {filename}"
        raise TrainError(message) from exc
    if not isinstance(parsed, dict):
        message = f"JSONL line {line_number} must be an object"
        raise TrainError(message)
    token_ids = parsed.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in token_ids
        )
    ):
        message = f"JSONL line {line_number} must contain integer token_ids"
        raise TrainError(message)
    return tuple(token_ids)


def _pad_or_truncate(ids: tuple[int, ...], sequence_length: int) -> tuple[int, ...]:
    """Keep the first ``sequence_length`` ids or pad on the right.

    Args:
        ids: One story's token ids.
        sequence_length: Target window length.

    Returns:
        A window of exactly ``sequence_length`` ids.
    """
    if len(ids) >= sequence_length:
        return ids[:sequence_length]
    return ids + (_PAD_ID,) * (sequence_length - len(ids))
