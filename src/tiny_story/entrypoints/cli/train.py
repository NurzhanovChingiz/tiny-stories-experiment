"""Train command for the LSTM language model."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import train_model
from tiny_story.entrypoints.cli.app import app

CONFIG_OPTION = typer.Option(
    Path("config/train.yaml"),
    "--config",
    help="YAML file that selects training.",
)
DATA_DIR_OPTION = typer.Option(
    None,
    "--data-dir",
    help="Directory of tokenized JSONL files. The default is the configured data_dir.",
)
CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    "--checkpoint-dir",
    help="Directory for the best checkpoint. The default is the configured checkpoint_dir.",
)
LOG_DIR_OPTION = typer.Option(
    None,
    "--log-dir",
    help="Directory for CSV Lightning logs. The default is the configured log_dir.",
)
LOG_PATH_OPTION = typer.Option(
    None,
    "--log-path",
    help=(
        "Override the run log file. "
        "The default is artifacts/train/logs/<dataset name>.log."
    ),
)


@app.command()
def train(
    config: Path = CONFIG_OPTION,
    data_dir: Path | None = DATA_DIR_OPTION,
    checkpoint_dir: Path | None = CHECKPOINT_DIR_OPTION,
    log_dir: Path | None = LOG_DIR_OPTION,
    log_path: Path | None = LOG_PATH_OPTION,
) -> None:
    """Train the configured LSTM on tokenized stories.

    Args:
        config: YAML training configuration.
        data_dir: Directory that replaces the configured tokenized directory.
        checkpoint_dir: Directory that replaces the configured checkpoint directory.
        log_dir: Directory that replaces the configured Lightning log directory.
        log_path: Run log file that replaces
            ``artifacts/train/logs/<dataset name>.log``.

    Returns:
        None.
    """
    outcome = train_model(
        config,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        log_path=log_path,
    )
    typer.echo(f"checkpoints {outcome.checkpoint_dir}")
    typer.echo(f"logs {outcome.log_dir}")
    if outcome.best_model_path is not None:
        typer.echo(f"saved {outcome.best_model_path}")
