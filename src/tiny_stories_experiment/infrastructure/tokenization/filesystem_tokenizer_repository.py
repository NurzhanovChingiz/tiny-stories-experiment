"""Filesystem adapter that stores one tokenizer JSON document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_stories_experiment.application.ports.tokenizer_repository import (
    TOKENIZER_FILENAME,
    TokenizerRepository,
)

if TYPE_CHECKING:
    from pathlib import Path


class FilesystemTokenizerRepository(TokenizerRepository):
    """Tokenizer documents stored as ``tokenizer.json`` under a named directory."""

    def save(self, destination: Path, name: str, content: bytes) -> Path:
        """Store ``tokenizer.json`` under ``destination/name`` and return its path.

        Args:
            destination: Directory that holds named tokenizer directories.
            name: Single path segment for this tokenizer.
            content: UTF-8 JSON document for ``tokenizer.json``.

        Returns:
            Path of the stored ``tokenizer.json``.

        Flow:
            1. Artifact directory — create the named directory when it is missing.
            2. Partial file — write the document beside the published file.
            3. Publish — replace ``tokenizer.json``, or remove the partial on failure.
        """
        # 1. Artifact directory
        directory = destination / name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / TOKENIZER_FILENAME
        partial = target.with_name(f"{TOKENIZER_FILENAME}.partial")

        # 2. Partial file
        completed = False
        try:
            partial.write_bytes(content)
            # 3. Publish
            partial.replace(target)
            completed = True
        finally:
            if not completed:
                partial.unlink(missing_ok=True)
        return target
