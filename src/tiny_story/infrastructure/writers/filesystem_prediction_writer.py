"""Filesystem prediction writer adapter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tiny_story.application.port.prediction_writer import JsonValue, PredictionWriter

if TYPE_CHECKING:
    from pathlib import Path


class FilesystemPredictionWriter(PredictionWriter):
    """Writes prediction results as JSON."""

    def __init__(self, output_dir: Path) -> None:
        """Store the directory that receives prediction files.

        Args:
            output_dir: Directory where named JSON files are written.

        Returns:
            None.
        """
        self._output_dir = output_dir

    def write(
        self,
        name: str,
        prediction: dict[str, JsonValue],
    ) -> None:
        """Write one prediction as a JSON file.

        Create the destination directory when needed, then store the payload
        as indented UTF-8 JSON named ``{name}.json``.

        Args:
            name: File stem for the prediction.
            prediction: JSON-compatible prediction payload.

        Returns:
            None.

        Flow:
            1. Ensure the output directory exists.
            2. Persist the named JSON file.
        """
        # 1. Ensure the output directory exists.
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Persist the named JSON file.
        output_path = self._output_dir / f"{name}.json"
        output_path.write_text(
            json.dumps(prediction, indent=2),
            encoding="utf-8",
        )
