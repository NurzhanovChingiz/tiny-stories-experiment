"""Download command for dataset acquisition."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import acquire_dataset
from tiny_story.entrypoints.cli.app import app

CONFIG_OPTION = typer.Option(
    Path("config/dataset.yaml"),
    help="YAML file that selects the Hugging Face dataset.",
)
RAW_DIR_OPTION = typer.Option(
    None,
    help="Override the destination directory from the configuration.",
)
CACHE_DIR_OPTION = typer.Option(
    None,
    help="Override the Hugging Face cache directory from the configuration.",
)
LOG_PATH_OPTION = typer.Option(
    None,
    help=(
        "Override the run log file. "
        "The default is artifacts/download/logs/<dataset name>.log."
    ),
)


@app.command()
def download(
    config: Path = CONFIG_OPTION,
    raw_dir: Path | None = RAW_DIR_OPTION,
    cache_dir: Path | None = CACHE_DIR_OPTION,
    log_path: Path | None = LOG_PATH_OPTION,
) -> None:
    """Acquire the configured dataset into raw storage.

    Args:
        config: YAML file that selects the Hugging Face dataset.
        raw_dir: Destination directory that replaces the configured one.
        cache_dir: Hugging Face cache directory that replaces the configured one.
        log_path: Run log file that replaces
            ``artifacts/download/logs/<dataset name>.log``.

    Returns:
        None.
    """
    outcome = acquire_dataset(
        config,
        destination=raw_dir,
        cache_dir=cache_dir,
        log_path=log_path,
    )
    for path in outcome.already_present_paths:
        typer.echo(f"present {path}")
    for path in outcome.written_paths:
        typer.echo(f"saved {path}")
