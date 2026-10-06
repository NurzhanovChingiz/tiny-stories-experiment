"""Prepare command for derived TinyStoriesV2 split files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from tiny_stories_experiment.composition import (
    DEFAULT_PROCESSED_DESTINATION,
    DEFAULT_RAW_DESTINATION,
    TINY_STORIES_V2,
    prepare_default_dataset,
)


def _as_path(value: Path) -> Path:
    """Return a path, including when the CLI supplied a string.

    Args:
        value: Directory path from the command option.

    Returns:
        That directory as a ``Path``.
    """
    return value if isinstance(value, Path) else Path(value)


def prepare_command(
    raw_directory: Annotated[
        Path,
        typer.Option(help="Directory of immutable TinyStoriesV2 JSONL files."),
    ] = DEFAULT_RAW_DESTINATION,
    processed_directory: Annotated[
        Path,
        typer.Option(help="Directory for derived train.jsonl and valid.jsonl files."),
    ] = DEFAULT_PROCESSED_DESTINATION,
) -> None:
    """Write derived TinyStoriesV2 JSONL files and leave the raw files unchanged.

    Args:
        raw_directory: Directory of immutable raw JSONL files.
        processed_directory: Directory for derived split files.

    Flow:
        1. Progress sink — plain INFO lines on stderr.
        2. Directory paths — accept a path or a string from the CLI.
        3. Prepare — write one derived JSONL file per raw split.
        4. Progress lines — log the source, each split count, and both directories.
    """
    # 1. Progress sink
    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")

    # 2. Directory paths
    raw_path = _as_path(raw_directory)
    processed_path = _as_path(processed_directory)

    # 3. Prepare
    report = prepare_default_dataset(raw_path, processed_path)

    # 4. Progress lines
    logger.info("dataset {}", TINY_STORIES_V2.source_url)
    for split in report.splits:
        logger.info("prepared {} {}", split.output_name, split.story_count)
    logger.info("raw directory {}", report.raw_directory)
    logger.info("processed directory {}", report.processed_directory)


def main() -> None:
    """Run the TinyStoriesV2 prepare command."""
    typer.run(prepare_command)
