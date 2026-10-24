"""Construct the Lightning trainer used for LSTM training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

if TYPE_CHECKING:
    from pathlib import Path


def build_trainer(
    *,
    accelerator: str,
    precision: str,
    max_epochs: int,
    min_delta: float,
    patience: int,
    checkpoint_dir: Path,
    log_dir: Path,
    enable_progress_bar: bool,
) -> Trainer:
    """Construct a trainer with CSV logging, early stopping, and one checkpoint.

    Purpose:
        Assemble the Lightning trainer that fits the language model.

    Strategy:
        Log with CSVLogger, stop when validation loss stalls, and keep the
        single best checkpoint.

    Args:
        accelerator: Lightning accelerator name, ``cpu`` or ``gpu``.
        precision: Lightning precision name.
        max_epochs: Maximum training epochs.
        min_delta: Early-stopping improvement threshold.
        patience: Epochs without improvement before stopping.
        checkpoint_dir: Directory for the best checkpoint.
        log_dir: Directory for CSV logs.
        enable_progress_bar: Whether Lightning draws a progress bar.

    Returns:
        Configured Lightning Trainer.

    Flow:
        1. Create the CSV logger.
        2. Construct the trainer with callbacks.
    """
    # 1. Create the CSV logger.
    csv_logger = CSVLogger(
        save_dir=str(log_dir.parent),
        name=log_dir.name,
        version="",
    )

    # 2. Construct the trainer with callbacks.
    precision_value: Any = precision
    return Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision_value,
        max_epochs=max_epochs,
        logger=csv_logger,
        enable_progress_bar=enable_progress_bar,
        callbacks=[
            EarlyStopping(
                min_delta=min_delta,
                monitor="val_loss",
                patience=patience,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch}-{val_loss:.6f}",
                dirpath=str(checkpoint_dir),
                verbose=False,
            ),
        ],
    )
