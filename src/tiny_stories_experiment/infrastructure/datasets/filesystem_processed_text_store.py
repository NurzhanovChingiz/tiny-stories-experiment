"""Filesystem adapter that writes one derived JSONL file per split."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tiny_stories_experiment.application.ports.processed_text_store import (
    ProcessedTextStore,
    processed_filename,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from tiny_stories_experiment.domain.dataset.text_sample import DatasetSplit


class FilesystemProcessedTextStore(ProcessedTextStore):
    """Derived story text stored as ordinary JSONL files."""

    def prepare(self, destination: Path) -> None:
        """Create the processed directory when it is missing.

        Args:
            destination: Directory that will hold derived JSONL files.
        """
        destination.mkdir(parents=True, exist_ok=True)

    def stage(
        self, destination: Path, split: DatasetSplit, texts: Iterable[str]
    ) -> int:
        """Write one split's stories to a partial file and leave the derived file.

        Args:
            destination: Directory that holds derived JSONL files.
            split: Split being staged.
            texts: Story text in source order.

        Returns:
            The number of stories written to the partial file.

        Flow:
            1. Partial file — write each story as one JSON line.
            2. Failed write — remove the partial file and leave the previous output.
        """
        partial = _partial_path(destination, split)
        completed = False
        count = 0
        try:
            # 1. Partial file
            with partial.open("w", encoding="utf-8") as handle:
                for text in texts:
                    line = json.dumps({"text": text}, ensure_ascii=False)
                    handle.write(f"{line}\n")
                    count += 1
            completed = True
        finally:
            # 2. Failed write
            if not completed:
                partial.unlink(missing_ok=True)
        return count

    def publish(self, destination: Path, splits: tuple[DatasetSplit, ...]) -> None:
        """Replace every derived file with its staged partial, or restore backups.

        Args:
            destination: Directory that holds derived JSONL files.
            splits: Staged splits to publish, in spec order.

        Flow:
            1. Replace outputs — move each partial into place and keep a backup.
            2. Failed replace — restore those backups and remove leftover partials.
            3. Drop backups — remove backups after every replace succeeds.
        """
        replaced: list[tuple[DatasetSplit, bool]] = []
        try:
            # 1. Replace outputs
            for split in splits:
                had_previous = _replace_staged(destination, split)
                replaced.append((split, had_previous))
        except OSError:
            # 2. Failed replace
            _restore_outputs(destination, tuple(replaced))
            self.discard(destination, splits)
            raise
        # 3. Drop backups
        for split in splits:
            _backup_path(destination, split).unlink(missing_ok=True)

    def discard(self, destination: Path, splits: tuple[DatasetSplit, ...]) -> None:
        """Remove staged partials and leave derived files in place.

        Args:
            destination: Directory that holds derived JSONL files.
            splits: Staged splits whose partial files should be removed.
        """
        for split in splits:
            _partial_path(destination, split).unlink(missing_ok=True)


def _partial_path(destination: Path, split: DatasetSplit) -> Path:
    """Return the partial path for one staged split.

    Args:
        destination: Directory that holds derived JSONL files.
        split: Split being staged.

    Returns:
        The ``*.jsonl.partial`` path for that split.
    """
    return destination / f"{processed_filename(split)}.partial"


def _output_path(destination: Path, split: DatasetSplit) -> Path:
    """Return the derived JSONL path for one split.

    Args:
        destination: Directory that holds derived JSONL files.
        split: Split being published.

    Returns:
        The ``train.jsonl`` or ``valid.jsonl`` path.
    """
    return destination / processed_filename(split)


def _backup_path(destination: Path, split: DatasetSplit) -> Path:
    """Return the backup path that holds the previous derived file.

    Args:
        destination: Directory that holds derived JSONL files.
        split: Split being published.

    Returns:
        The ``*.jsonl.previous`` path for that split.
    """
    return destination / f"{processed_filename(split)}.previous"


def _replace_staged(destination: Path, split: DatasetSplit) -> bool:
    """Move one partial into place, restoring the previous file if that move fails.

    Args:
        destination: Directory that holds derived JSONL files.
        split: Staged split to publish.

    Returns:
        True when a previous derived file was moved aside.

    Flow:
        1. Backup — move the current derived file aside when it exists.
        2. Partial replace — move the partial into that derived name.
        3. Restore one — put the backup back when the partial move fails.
    """
    partial = _partial_path(destination, split)
    output = _output_path(destination, split)
    backup = _backup_path(destination, split)
    # 1. Backup
    had_previous = output.is_file()
    if had_previous:
        output.replace(backup)
    try:
        # 2. Partial replace
        partial.replace(output)
    except OSError:
        # 3. Restore one
        if backup.is_file():
            backup.replace(output)
        raise
    return had_previous


def _restore_outputs(
    destination: Path, replaced: tuple[tuple[DatasetSplit, bool], ...]
) -> None:
    """Return derived files to the bytes they had before this publish.

    Args:
        destination: Directory that holds derived JSONL files.
        replaced: Splits already replaced, with whether each had a previous file.
    """
    for split, had_previous in reversed(replaced):
        output = _output_path(destination, split)
        backup = _backup_path(destination, split)
        if had_previous and backup.is_file():
            backup.replace(output)
        elif not had_previous:
            output.unlink(missing_ok=True)
