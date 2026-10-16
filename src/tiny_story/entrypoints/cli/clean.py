"""Clean command for interim story text."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import clean_dataset
from tiny_story.entrypoints.cli.app import app

DATASET_CONFIG_OPTION = typer.Option(
    Path("config/dataset.yaml"),
    "--dataset-config",
    help="YAML file that names the raw dataset files.",
)
CONFIG_OPTION = typer.Option(
    Path("config/clean_text.yaml"),
    "--config",
    help="YAML file that selects text cleaning.",
)
OUTPUT_DIR_OPTION = typer.Option(
    None,
    "--output-dir",
    help="Directory for cleaned JSONL files. The default is data/interim/<dataset name>.",
)
LOG_PATH_OPTION = typer.Option(
    None,
    "--log-path",
    help=(
        "Override the run log file. "
        "The default is artifacts/clean/logs/<dataset name>.log."
    ),
)


@app.command()
def clean(
    dataset_config: Path = DATASET_CONFIG_OPTION,
    config: Path = CONFIG_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
    log_path: Path | None = LOG_PATH_OPTION,
) -> None:
    """Clean raw story files into the interim directory.

    Args:
        dataset_config: YAML file that names the raw dataset files.
        config: YAML file that selects text cleaning.
        output_dir: Directory that replaces ``data/interim/<dataset name>``.
        log_path: Run log file that replaces
            ``artifacts/clean/logs/<dataset name>.log``.

    Returns:
        None.
    """
    written = clean_dataset(
        dataset_config,
        config,
        output_dir=output_dir,
        log_path=log_path,
    )
    for path in written:
        typer.echo(f"saved {path}")
