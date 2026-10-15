"""Command line for dataset acquisition."""

from __future__ import annotations

from pathlib import Path

import typer

from tiny_story.composition import acquire_dataset

app = typer.Typer(add_completion=False, no_args_is_help=True)
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


@app.callback()
def main() -> None:
    """Keep dataset commands grouped under this program.

    Typer would otherwise collapse the only command into the program itself.
    ``download`` stays a subcommand so ``make download`` can call it by name.

    Returns:
        None.
    """


@app.command()
def download(
    config: Path = CONFIG_OPTION,
    raw_dir: Path | None = RAW_DIR_OPTION,
    cache_dir: Path | None = CACHE_DIR_OPTION,
) -> None:
    """Acquire the configured dataset into raw storage.

    Args:
        config: YAML file that selects the Hugging Face dataset.
        raw_dir: Destination directory that replaces the configured one.
        cache_dir: Hugging Face cache directory that replaces the configured one.

    Returns:
        None.
    """
    outcome = acquire_dataset(config, destination=raw_dir, cache_dir=cache_dir)
    for path in outcome.already_present_paths:
        typer.echo(f"present {path}")
    for path in outcome.written_paths:
        typer.echo(f"saved {path}")


if __name__ == "__main__":
    app()
