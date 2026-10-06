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

    def write(
        self, destination: Path, split: DatasetSplit, texts: Iterable[str]
    ) -> int:
        """Replace one split's derived JSONL with the given stories.

        Args:
            destination: Directory that holds derived JSONL files.
            split: Split being written.
            texts: Story text in source order.

        Returns:
            The number of stories written.

        Flow:
            1. Partial file — write each story as one JSON line.
            2. Completed replace — move that file into place only after every story.
            3. Failed write — remove the partial file and leave the previous output.
        """
        output = destination / processed_filename(split)
        partial = destination / f"{processed_filename(split)}.partial"
        completed = False
        count = 0
        try:
            # 1. Partial file
            with partial.open("w", encoding="utf-8") as handle:
                for text in texts:
                    line = json.dumps({"text": text}, ensure_ascii=False)
                    handle.write(f"{line}\n")
                    count += 1
            # 2. Completed replace
            partial.replace(output)
            completed = True
        finally:
            # 3. Failed write
            if not completed:
                partial.unlink(missing_ok=True)
        return count
