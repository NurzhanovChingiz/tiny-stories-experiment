"""Run one Lightning training job from stored configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from tiny_story.application.results.train_outcome import TrainOutcome
from tiny_story.infrastructure.lightning.dataset import (
    TokenizedStoryDataset,
    load_token_id_sequences,
    load_vocabulary_size,
    window_sequences,
)
from tiny_story.infrastructure.lightning.module import CausalLmLightningModule
from tiny_story.infrastructure.lightning.runtime import resolve_accelerator, seed_run
from tiny_story.infrastructure.lightning.split import split_sequences
from tiny_story.infrastructure.lightning.trainer_factory import build_trainer

if TYPE_CHECKING:
    from tiny_story.infrastructure.config.train_config import TrainConfig


class LightningTrainingRun:
    """Fit the configured LSTM on tokenized stories.

    Args:
        config: Validated training settings.
        data_dir: Tokenized JSONL directory that replaces ``config.data_dir``.
        checkpoint_dir: Checkpoint directory that replaces ``config.checkpoint_dir``.
        log_dir: Lightning log directory that replaces ``config.log_dir``.
    """

    def __init__(
        self,
        config: TrainConfig,
        data_dir: Path,
        checkpoint_dir: Path,
        log_dir: Path,
    ) -> None:
        """Store the configuration and resolved artifact paths.

        Args:
            config: Validated training settings.
            data_dir: Tokenized JSONL directory.
            checkpoint_dir: Checkpoint directory.
            log_dir: Lightning log directory.

        Returns:
            None.
        """
        self._config = config
        self._data_dir = data_dir
        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir

    def fit(self) -> TrainOutcome:
        """Run dataset, module, and trainer stages and fit the model.

        Purpose:
            Execute the training workflow from tokenized stories through
            ``trainer.fit``.

        Strategy:
            Seed the run, load and split windows, build loaders and the
            Lightning objects, fit, and return the artifact paths.

        Returns:
            Checkpoint and log directories, plus the best checkpoint path.

        Flow:
            1. Seed generators and resolve the accelerator.
            2. Load vocabulary size and token-id sequences.
            3. Split and window the sequences.
            4. Build dataloaders.
            5. Build the Lightning module and trainer.
            6. Fit the model.
            7. Return the written artifact paths.
        """
        # 1. Seed generators and resolve the accelerator.
        seed_run(self._config.seed)
        accelerator = resolve_accelerator()

        # 2. Load vocabulary size and token-id sequences.
        vocab_path = self._vocab_path()
        vocab_size = load_vocabulary_size(vocab_path)
        sequences = load_token_id_sequences(self._data_dir)

        # 3. Split and window the sequences.
        train, validation, _test = split_sequences(
            sequences,
            test_size=self._config.split.test_size,
            val_size=self._config.split.val_size,
            seed=self._config.seed,
        )
        train_windows = window_sequences(train, self._config.sequence_length)
        val_windows = window_sequences(validation, self._config.sequence_length)

        # 4. Build dataloaders.
        train_loader = self._dataloader(train_windows, shuffle=True)
        val_loader = self._dataloader(val_windows, shuffle=False)

        # 5. Build the Lightning module and trainer.
        steps_per_epoch = max(len(train_loader), 1)
        module = self._module(vocab_size, steps_per_epoch)
        trainer = build_trainer(
            accelerator=accelerator,
            precision=self._precision(),
            max_epochs=self._config.epochs,
            min_delta=self._config.min_delta,
            patience=self._config.patience,
            checkpoint_dir=self._checkpoint_dir,
            log_dir=self._log_dir,
            enable_progress_bar=self._config.enable_progress_bar,
        )

        # 6. Fit the model.
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # 7. Return the written artifact paths.
        return TrainOutcome(
            checkpoint_dir=self._checkpoint_dir,
            log_dir=self._log_dir,
            best_model_path=_best_checkpoint_path(trainer),
        )

    def _vocab_path(self) -> Path:
        """Resolve the vocabulary file under the data directory when relative.

        Returns:
            Absolute or data-directory-relative vocabulary path.
        """
        if self._config.vocab_file.is_absolute():
            return self._config.vocab_file
        return self._data_dir / self._config.vocab_file

    def _precision(self) -> str:
        """Return mixed precision when AMP is on, otherwise full precision.

        Returns:
            Lightning precision name.
        """
        if self._config.amp:
            return self._config.precision
        return "32-true"

    def _dataloader(
        self,
        windows: tuple[tuple[int, ...], ...],
        *,
        shuffle: bool,
    ) -> DataLoader[object]:
        """Wrap windows in a DataLoader using the stored loader settings.

        Args:
            windows: Equal-length token-id windows.
            shuffle: Whether to shuffle samples each epoch.

        Returns:
            DataLoader over ``windows``.
        """
        workers = self._config.num_workers
        return DataLoader(
            TokenizedStoryDataset(windows),
            batch_size=self._config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self._config.pin_memory,
            persistent_workers=self._config.persistent_workers and workers > 0,
        )

    def _module(self, vocab_size: int, steps_per_epoch: int) -> CausalLmLightningModule:
        """Construct the Lightning module from stored hyperparameters.

        Args:
            vocab_size: Vocabulary width from ``vocab.json``.
            steps_per_epoch: Optimizer steps per epoch.

        Returns:
            Configured ``CausalLmLightningModule``.
        """
        return CausalLmLightningModule(
            vocab_size,
            self._config.model,
            learning_rate=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
            scheduler_type=self._config.scheduler_type,
            epochs=self._config.epochs,
            warmup_epochs=self._config.warmup_epochs,
            eta_min=self._config.eta_min,
            step_size=self._config.step_size,
            gamma=self._config.gamma,
            max_lr=self._config.max_lr,
            steps_per_epoch=steps_per_epoch,
        )


def _best_checkpoint_path(trainer: object) -> Path | None:
    """Return the trainer's best checkpoint path when one was written.

    Args:
        trainer: Fitted Lightning trainer.

    Returns:
        The best checkpoint file, or None when Lightning stored no path.
    """
    callback = getattr(trainer, "checkpoint_callback", None)
    raw = getattr(callback, "best_model_path", "")
    if not raw:
        return None
    return Path(raw)
