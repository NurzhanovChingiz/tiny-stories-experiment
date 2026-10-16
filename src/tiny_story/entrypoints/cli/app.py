"""Shared command-line application."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
