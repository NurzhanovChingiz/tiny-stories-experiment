"""Tokenize command for processed story text."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import tokenize_dataset
from tiny_story.entrypoints.cli.app import app

DATASET_CONFIG_OPTION = typer.Option(
    Path("config/dataset.yaml"),
    "--dataset-config",
    help="YAML file that names the dataset files.",
)
CONFIG_OPTION = typer.Option(
    Path("config/tokenize.yaml"),
    "--config",
    help="YAML file that selects tokenization.",
)
INPUT_DIR_OPTION = typer.Option(
    None,
    "--input-dir",
    help="Directory of normalized JSONL files. The default is data/processed/<dataset name>.",
)
OUTPUT_DIR_OPTION = typer.Option(
    None,
    "--output-dir",
    help="Directory for tokenized JSONL files. The default is data/tokenized/<dataset name>.",
)
LOG_PATH_OPTION = typer.Option(
    None,
    "--log-path",
    help=(
        "Override the run log file. "
        "The default is artifacts/tokenize/logs/<dataset name>.log."
    ),
)


@app.command()
def tokenize(
    dataset_config: Path = DATASET_CONFIG_OPTION,
    config: Path = CONFIG_OPTION,
    input_dir: Path | None = INPUT_DIR_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
    log_path: Path | None = LOG_PATH_OPTION,
) -> None:
    """Tokenize normalized story files into the tokenized directory.

    Args:
        dataset_config: YAML file that names the dataset files.
        config: YAML file that selects tokenization.
        input_dir: Directory that replaces ``data/processed/<dataset name>``.
        output_dir: Directory that replaces ``data/tokenized/<dataset name>``.
        log_path: Run log file that replaces
            ``artifacts/tokenize/logs/<dataset name>.log``.

    Returns:
        None.
    """
    written = tokenize_dataset(
        dataset_config,
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        log_path=log_path,
    )
    for path in written:
        typer.echo(f"saved {path}")
