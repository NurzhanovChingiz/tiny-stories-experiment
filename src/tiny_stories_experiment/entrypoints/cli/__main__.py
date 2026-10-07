"""Run the tiny stories command-line interface."""

import typer

from tiny_stories_experiment.entrypoints.cli.commands.download import download_command
from tiny_stories_experiment.entrypoints.cli.commands.prepare import prepare_command
from tiny_stories_experiment.entrypoints.cli.commands.train_tokenizer import (
    train_tokenizer_command,
)

app = typer.Typer(no_args_is_help=True)
app.command("download")(download_command)
app.command("prepare")(prepare_command)
app.command("train-tokenizer")(train_tokenizer_command)


def main() -> None:
    """Run the download, prepare, and train-tokenizer commands."""
    app()


if __name__ == "__main__":
    main()
