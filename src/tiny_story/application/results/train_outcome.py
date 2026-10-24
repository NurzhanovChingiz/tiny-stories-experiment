"""Result of one Lightning training run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """Directories written by one training run.

    Attributes:
        checkpoint_dir: Directory that stores the best checkpoint.
        log_dir: Directory that stores CSV Lightning logs.
        best_model_path: Best checkpoint file, or None when none was saved.
    """

    checkpoint_dir: Path
    log_dir: Path
    best_model_path: Path | None
