"""Run the tiny stories command-line interface."""

import typer

from tiny_stories_experiment.entrypoints.cli.commands.download import download_command
from tiny_stories_experiment.entrypoints.cli.commands.prepare import prepare_command

app = typer.Typer(no_args_is_help=True)
app.command("download")(download_command)
app.command("prepare")(prepare_command)


def main() -> None:
    """Run the download and prepare commands."""
    app()


if __name__ == "__main__":
    main()
