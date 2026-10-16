"""Normalize command for derived story text."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import normalize_dataset
from tiny_story.entrypoints.cli.app import app

DATASET_CONFIG_OPTION = typer.Option(
    Path("config/dataset.yaml"),
    "--dataset-config",
    help="YAML file that names the raw dataset files.",
)
CONFIG_OPTION = typer.Option(
    Path("config/normalize_text.yaml"),
    "--config",
    help="YAML file that selects text normalization.",
)
INPUT_DIR_OPTION = typer.Option(
    None,
    "--input-dir",
    help="Directory of cleaned JSONL files. The default is data/interim/<dataset name>.",
)
OUTPUT_DIR_OPTION = typer.Option(
    None,
    "--output-dir",
    help="Directory for normalized JSONL files. The default is data/processed/<dataset name>.",
)
LOG_PATH_OPTION = typer.Option(
    None,
    "--log-path",
    help=(
        "Override the run log file. "
        "The default is artifacts/normalize/logs/<dataset name>.log."
    ),
)


@app.command()
def normalize(
    dataset_config: Path = DATASET_CONFIG_OPTION,
    config: Path = CONFIG_OPTION,
    input_dir: Path | None = INPUT_DIR_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
    log_path: Path | None = LOG_PATH_OPTION,
) -> None:
    """Normalize cleaned story files into the processed directory.

    Args:
        dataset_config: YAML file that names the dataset files.
        config: YAML file that selects text normalization.
        input_dir: Directory that replaces ``data/interim/<dataset name>``.
        output_dir: Directory that replaces ``data/processed/<dataset name>``.
        log_path: Run log file that replaces
            ``artifacts/normalize/logs/<dataset name>.log``.

    Returns:
        None.
    """
    written = normalize_dataset(
        dataset_config,
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        log_path=log_path,
    )
    for path in written:
        typer.echo(f"saved {path}")
