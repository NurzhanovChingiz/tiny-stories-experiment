"""Train a tokenizer on the prepared TinyStories splits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from tiny_stories_experiment.composition import (
    DEFAULT_PROCESSED_DESTINATION,
    DEFAULT_TOKENIZER_DESTINATION,
    DEFAULT_TOKENIZER_SPEC,
    train_default_tokenizer,
)


def _as_path(value: Path) -> Path:
    """Return a path, including when the CLI supplied a string.

    Args:
        value: Directory path from the command option.

    Returns:
        That directory as a ``Path``.
    """
    return value if isinstance(value, Path) else Path(value)


def train_tokenizer_command(
    processed_directory: Annotated[
        Path,
        typer.Option(help="Directory of derived train.jsonl and valid.jsonl files."),
    ] = DEFAULT_PROCESSED_DESTINATION,
    artifact_directory: Annotated[
        Path,
        typer.Option(help="Directory for the trained tokenizer document."),
    ] = DEFAULT_TOKENIZER_DESTINATION,
    name: Annotated[
        str,
        typer.Option(help="Artifact directory name under the tokenizer directory."),
    ] = DEFAULT_TOKENIZER_SPEC.name,
    vocab_size: Annotated[
        int,
        typer.Option(help="Maximum vocabulary size, including the end-of-text token."),
    ] = DEFAULT_TOKENIZER_SPEC.vocab_size,
    min_frequency: Annotated[
        int,
        typer.Option(help="Smallest pair count that may become a merge."),
    ] = DEFAULT_TOKENIZER_SPEC.min_frequency,
) -> None:
    """Train a BPE tokenizer on the prepared splits and store tokenizer.json.

    Prepared JSONL files are read and left at the same byte size.

    Args:
        processed_directory: Directory of derived ``train.jsonl`` and
            ``valid.jsonl`` files.
        artifact_directory: Directory for ``name/tokenizer.json``.
        name: Artifact directory name under ``artifact_directory``.
        vocab_size: Maximum vocabulary size, including the end-of-text token.
        min_frequency: Smallest pair count that may become a merge.

    Flow:
        1. Progress sink — plain INFO lines on stderr.
        2. Directory paths — accept a path or a string from the CLI.
        3. Train — fit BPE on both prepared splits and store the document.
        4. Progress lines — log the name, vocab size, story count, and paths.
    """
    # 1. Progress sink
    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")

    # 2. Directory paths
    processed_path = _as_path(processed_directory)
    artifact_path = _as_path(artifact_directory)

    # 3. Train
    report = train_default_tokenizer(
        processed_path,
        artifact_path,
        name=name,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    # 4. Progress lines
    logger.info("tokenizer {}", name)
    logger.info("vocab {}", report.vocab_size)
    logger.info("stories {}", report.story_count)
    logger.info("artifact {}", report.artifact_path)
    logger.info("processed directory {}", report.processed_directory)


def main() -> None:
    """Run the TinyStories tokenizer training command."""
    typer.run(train_tokenizer_command)
