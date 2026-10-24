"""Seed generators and choose a Lightning accelerator."""

from __future__ import annotations

import random

import torch
from lightning.pytorch import seed_everything


def seed_run(seed: int) -> None:
    """Seed Python, PyTorch, and Lightning random number generators.

    Args:
        seed: Value applied to every seeded generator.

    Returns:
        None.
    """
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def resolve_accelerator() -> str:
    """Return ``gpu`` when CUDA is visible, otherwise ``cpu``.

    Returns:
        Lightning accelerator name.
    """
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"
