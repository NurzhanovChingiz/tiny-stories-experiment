"""Download command for the published TinyStoriesV2 files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from tiny_stories_experiment.composition import (
    DEFAULT_RAW_DESTINATION,
    TINY_STORIES_V2,
    download_default_dataset,
)


def download_command(
    destination: Annotated[
        Path,
        typer.Option(help="Directory for the immutable TinyStoriesV2 JSONL files."),
    ] = DEFAULT_RAW_DESTINATION,
) -> None:
    """Download TinyStoriesV2 JSONL files into the raw data directory.

    Args:
        destination: Directory for the raw JSONL files.

    Flow:
        1. Progress sink — plain INFO lines on stderr.
        2. Destination path — accept a path or a string from the CLI.
        3. Download — place TinyStoriesV2 in that directory.
        4. Progress lines — log the source, kept files, downloads, and directory.
    """
    # 1. Progress sink
    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")

    # 2. Destination path
    raw_directory = destination if isinstance(destination, Path) else Path(destination)

    # 3. Download
    report = download_default_dataset(raw_directory)

    # 4. Progress lines
    logger.info("dataset {}", TINY_STORIES_V2.source_url)
    for filename in report.kept:
        logger.info("kept {}", filename)
    for filename in report.downloaded:
        logger.info("downloaded {}", filename)
    logger.info("raw directory {}", report.destination)


def main() -> None:
    """Run the TinyStoriesV2 raw download command."""
    typer.run(download_command)
