"""Shared command-line application."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _keep_subcommands() -> None:
    """Keep download, clean, normalize, tokenize, and train as subcommands.

    Typer would otherwise collapse a single imported command into the program
    itself, so ``train`` would reject the ``train`` argument.

    Returns:
        None.
    """
