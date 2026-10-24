"""Per-split next-token loss, perplexity, and top-k accuracy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

_PAD_ID = 0


class LanguageModelMetrics:
    """Accumulate next-token metrics for one train, validation, or test split.

    Args:
        vocab_size: Vocabulary width used to cap top-5 at the class count.
    """

    def __init__(self, vocab_size: int) -> None:
        """Start an empty tracker for one split.

        Args:
            vocab_size: Vocabulary width used to cap top-5 at the class count.

        Returns:
            None.
        """
        self._topk = min(5, vocab_size)
        self._loss_sum = 0.0
        self._loss_count = 0
        self._top1 = 0
        self._topk_hits = 0
        self._token_count = 0

    def update(self, logits: Tensor, targets: Tensor, loss: Tensor) -> None:
        """Add one batch of next-token scores, ignoring pad targets.

        Args:
            logits: Predicted scores of shape ``(batch, time, vocab)``.
            targets: Integer next-token ids of shape ``(batch, time)``.
            loss: Scalar batch loss already reduced by the caller.

        Returns:
            None.

        Flow:
            1. Record the batch loss.
            2. Score non-pad tokens for top-1 and top-k hits.
        """
        # 1. Record the batch loss.
        self._loss_sum += float(loss.detach().item())
        self._loss_count += 1

        # 2. Score non-pad tokens for top-1 and top-k hits.
        flat_logits = logits.detach().reshape(-1, logits.size(-1))
        flat_targets = targets.detach().reshape(-1)
        mask = flat_targets != _PAD_ID
        if not mask.any():
            return
        selected = flat_logits[mask]
        labels = flat_targets[mask]
        ranking = selected.topk(self._topk, dim=-1).indices
        self._top1 += int((ranking[:, 0] == labels).sum().item())
        self._topk_hits += int(
            (ranking == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        )
        self._token_count += int(mask.sum().item())

    def compute(self) -> dict[str, float]:
        """Return epoch averages for the recorded batches.

        Returns:
            ``loss``, ``perplexity``, ``acc_top1``, and ``acc_top5``.
        """
        average_loss = self._loss_sum / max(self._loss_count, 1)
        perplexity = math.exp(average_loss) if average_loss < 20.0 else math.inf
        tokens = max(self._token_count, 1)
        return {
            "loss": average_loss,
            "perplexity": perplexity,
            "acc_top1": self._top1 / tokens,
            "acc_top5": self._topk_hits / tokens,
        }

    def reset(self) -> None:
        """Clear recorded batches so the next epoch starts empty.

        Returns:
            None.
        """
        self._loss_sum = 0.0
        self._loss_count = 0
        self._top1 = 0
        self._topk_hits = 0
        self._token_count = 0
