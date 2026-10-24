"""Lightning module that trains the LSTM language model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import AdamW

from tiny_story.infrastructure.lightning.lstm import TinyLstmLanguageModel
from tiny_story.infrastructure.lightning.metrics import LanguageModelMetrics
from tiny_story.infrastructure.lightning.scheduler import build_scheduler

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
    from torch import Tensor

    from tiny_story.domain.entities.language_model import CausalLmSpec

_PAD_ID = 0


class CausalLmLightningModule(LightningModule):
    """Train an LSTM language model with next-token loss and split metrics.

    Args:
        vocab_size: Vocabulary width, including special tokens.
        spec: LSTM width settings.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        scheduler_type: Lightning scheduler name.
        epochs: Total training epochs.
        warmup_epochs: Warmup length for cosine-warmup.
        eta_min: Minimum cosine learning rate.
        step_size: StepLR period.
        gamma: StepLR decay.
        max_lr: OneCycleLR peak learning rate.
        steps_per_epoch: Optimizer steps per epoch for OneCycleLR.
    """

    def __init__(
        self,
        vocab_size: int,
        spec: CausalLmSpec,
        *,
        learning_rate: float,
        weight_decay: float,
        scheduler_type: str,
        epochs: int,
        warmup_epochs: int,
        eta_min: float,
        step_size: int,
        gamma: float,
        max_lr: float,
        steps_per_epoch: int,
    ) -> None:
        """Build the network, loss, and per-split metric trackers from numbers.

        Purpose:
            Bind LSTM training hyperparameters and construct the objects this
            module owns.

        Strategy:
            Store the optimizer and scheduler numbers, construct the LSTM and
            pad-ignoring loss, then create independent train, validation, and
            test trackers.

        Args:
            vocab_size: Vocabulary width, including special tokens.
            spec: LSTM width settings.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW weight decay.
            scheduler_type: Lightning scheduler name.
            epochs: Total training epochs.
            warmup_epochs: Warmup length for cosine-warmup.
            eta_min: Minimum cosine learning rate.
            step_size: StepLR period.
            gamma: StepLR decay.
            max_lr: OneCycleLR peak learning rate.
            steps_per_epoch: Optimizer steps per epoch for OneCycleLR.

        Returns:
            None.

        Flow:
            1. Persist hyperparameters.
            2. Construct the network and loss.
            3. Construct three split trackers.
        """
        super().__init__()
        # 1. Persist hyperparameters.
        self.save_hyperparameters()
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._scheduler_type = scheduler_type
        self._epochs = epochs
        self._warmup_epochs = warmup_epochs
        self._eta_min = eta_min
        self._step_size = step_size
        self._gamma = gamma
        self._max_lr = max_lr
        self._steps_per_epoch = steps_per_epoch

        # 2. Construct the network and loss.
        self._network = TinyLstmLanguageModel(
            vocab_size,
            spec,
            pad_id=_PAD_ID,
        )
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=_PAD_ID)

        # 3. Construct three split trackers.
        self._train_metrics = LanguageModelMetrics(vocab_size)
        self._val_metrics = LanguageModelMetrics(vocab_size)
        self._test_metrics = LanguageModelMetrics(vocab_size)

    def _shared_step(self, batch: Tensor, tracker: LanguageModelMetrics) -> Tensor:
        """Run next-token prediction and update one split's metrics.

        Purpose:
            Score one window batch and record loss and top-k hits.

        Strategy:
            Predict logits for every prefix token, compare them to the shifted
            targets, ignore pad ids, and update the given tracker.

        Args:
            batch: Token-id windows of shape ``(batch, time)``.
            tracker: Metric tracker for the current split.

        Returns:
            Scalar batch loss.

        Flow:
            1. Predict next-token logits.
            2. Compute pad-ignoring cross-entropy.
            3. Update the split tracker.
        """
        # 1. Predict next-token logits.
        logits = self._network(batch[:, :-1])
        targets = batch[:, 1:]

        # 2. Compute pad-ignoring cross-entropy.
        loss = self._loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        # 3. Update the split tracker.
        tracker.update(logits, targets, loss)
        return loss

    def _log_epoch_metrics(self, tracker: LanguageModelMetrics, split: str) -> None:
        """Publish one split's epoch metrics, then reset the tracker.

        Args:
            tracker: Metric tracker that received updates this epoch.
            split: Log-name prefix such as ``train`` or ``val``.

        Returns:
            None.
        """
        scores = tracker.compute()
        if getattr(self, "_trainer", None) is not None:
            for name, value in scores.items():
                self.log(
                    f"{split}_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name in {"loss", "perplexity"},
                    logger=True,
                )
        tracker.reset()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Compute training loss and update training metrics for one batch.

        Args:
            batch: Token-id windows.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar training loss.
        """
        del batch_idx
        loss = self._shared_step(batch, self._train_metrics)
        if getattr(self, "_trainer", None) is not None:
            self.log(
                "train_loss_step",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Compute validation loss and update validation metrics.

        Args:
            batch: Token-id windows.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar validation loss.
        """
        del batch_idx
        return self._shared_step(batch, self._val_metrics)

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Compute test loss and update test metrics.

        Args:
            batch: Token-id windows.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar test loss.
        """
        del batch_idx
        return self._shared_step(batch, self._test_metrics)

    def on_train_epoch_end(self) -> None:
        """Log and reset training metrics after an epoch."""
        self._log_epoch_metrics(self._train_metrics, "train")

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics after an epoch."""
        self._log_epoch_metrics(self._val_metrics, "val")

    def on_test_epoch_end(self) -> None:
        """Log and reset test metrics after an epoch."""
        self._log_epoch_metrics(self._test_metrics, "test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Construct AdamW and the configured learning-rate scheduler.

        Returns:
            Lightning optimizer and scheduler mapping.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=self._scheduler_type,
            epochs=self._epochs,
            warmup_epochs=self._warmup_epochs,
            eta_min=self._eta_min,
            step_size=self._step_size,
            gamma=self._gamma,
            max_lr=self._max_lr,
            steps_per_epoch=self._steps_per_epoch,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
