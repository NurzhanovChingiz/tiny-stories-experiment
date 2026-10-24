"""Build Lightning-compatible learning-rate scheduler configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
    StepLR,
)

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import LRSchedulerConfigType
    from torch.optim import Optimizer


def build_scheduler(
    optimizer: Optimizer,
    *,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
    step_size: int,
    gamma: float,
    max_lr: float,
    steps_per_epoch: int | None = None,
) -> LRSchedulerConfigType:
    """Build a learning-rate scheduler and return a Lightning-compatible dict.

    Purpose:
        Map a caller-selected scheduler name onto a Lightning dict the trainer
        can consume.

    Strategy:
        Normalize the type name, then construct the matching PyTorch scheduler
        and wrap it with Lightning interval and frequency, failing closed on
        an unknown name.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        scheduler_type: One of cosine_warmup, cosine, step, or onecycle.
        epochs: Total training epochs.
        warmup_epochs: Warmup length for ``cosine_warmup``.
        eta_min: Minimum learning rate for cosine schedules.
        step_size: Epoch period for StepLR.
        gamma: StepLR multiplicative decay.
        max_lr: Peak learning rate for OneCycleLR.
        steps_per_epoch: Optimizer steps per epoch, required by OneCycleLR.

    Returns:
        Lightning scheduler dict with ``scheduler``, ``interval``, and
        ``frequency``.

    Raises:
        ValueError: The scheduler name is unknown, or OneCycleLR is missing
            ``steps_per_epoch``.

    Flow:
        1. Normalize the scheduler type.
        2. Construct and return the requested Lightning scheduler dict.
    """
    # 1. Normalize the scheduler type.
    name = scheduler_type.lower()

    # 2. Construct and return the requested Lightning scheduler dict.
    if name == "cosine_warmup":
        warmup = LinearLR(optimizer, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        return {
            "scheduler": SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            ),
            "interval": "epoch",
            "frequency": 1,
        }
    if name == "cosine":
        return {
            "scheduler": CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min),
            "interval": "epoch",
            "frequency": 1,
        }
    if name == "step":
        return {
            "scheduler": StepLR(optimizer, step_size=step_size, gamma=gamma),
            "interval": "epoch",
            "frequency": 1,
        }
    if name == "onecycle":
        if steps_per_epoch is None:
            message = "steps_per_epoch is required for onecycle"
            raise ValueError(message)
        return {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            ),
            "interval": "step",
            "frequency": 1,
        }
    message = (
        f"Unknown scheduler_type '{scheduler_type}'. "
        "Choose from: cosine_warmup, cosine, step, onecycle."
    )
    raise ValueError(message)
