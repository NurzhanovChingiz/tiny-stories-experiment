"""Seeded train, validation, and test split of token-id sequences."""

from __future__ import annotations

from tiny_story.domain.errors import TrainError


def split_sequences(
    sequences: tuple[tuple[int, ...], ...],
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    """Shuffle sequences and carve test, validation, and train slices.

    Purpose:
        Produce three disjoint splits from one collection of windows.

    Strategy:
        Shuffle a copy with a local RNG, take the test slice, then the
        validation slice, and leave the remainder for training.

    Args:
        sequences: Windows to split.
        test_size: Fraction assigned to test.
        val_size: Fraction assigned to validation.
        seed: Shuffle seed.

    Returns:
        Train, validation, and test sequence tuples.

    Raises:
        TrainError: Training or validation would be empty.

    Flow:
        1. Shuffle a copy of the sequences.
        2. Carve test, validation, and train slices.
        3. Reject an empty train or validation split.
    """
    # 1. Shuffle a copy of the sequences.
    items = list(sequences)
    _shuffle(items, seed)

    # 2. Carve test, validation, and train slices.
    test_count = int(len(items) * test_size)
    val_count = int(len(items) * val_size)
    test = tuple(items[:test_count])
    validation = tuple(items[test_count : test_count + val_count])
    train = tuple(items[test_count + val_count :])

    # 3. Reject an empty train or validation split.
    if not train or not validation:
        message = (
            "Training needs at least one train sequence and one validation "
            "sequence after the configured split"
        )
        raise TrainError(message)
    return train, validation, test


def _shuffle(items: list[tuple[int, ...]], seed: int) -> None:
    """Shuffle ``items`` in place with a seeded linear congruential generator.

    Args:
        items: Sequences to reorder.
        seed: Generator seed.

    Returns:
        None.
    """
    state = seed % 2147483647
    for index in range(len(items) - 1, 0, -1):
        state = (state * 1103515245 + 12345) % 2147483647
        swap = state % (index + 1)
        items[index], items[swap] = items[swap], items[index]
