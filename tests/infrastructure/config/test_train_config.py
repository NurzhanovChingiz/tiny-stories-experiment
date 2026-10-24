"""Tests for training configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_story.domain.errors import InvalidTrainConfigError
from tiny_story.infrastructure.config.train_config import load_train_config

_VALID = """\
name: tiny_stories
seed: 137
data_dir: data/tokenized/tiny_stories
vocab_file: vocab.json
sequence_length: 64
batch_size: 8
epochs: 3
learning_rate: 0.003
weight_decay: 0.01
scheduler_type: cosine_warmup
warmup_epochs: 1
eta_min: 1.0e-6
step_size: 10
gamma: 0.1
max_lr: 0.003
min_delta: 0.001
patience: 5
num_workers: 0
pin_memory: false
persistent_workers: false
amp: false
precision: 32-true
enable_progress_bar: false
model:
  embedding_dim: 64
  hidden_size: 128
  num_layers: 1
  dropout: 0.1
split:
  test_size: 0.2
  val_size: 0.15
checkpoint_dir: artifacts/train/checkpoints
log_dir: artifacts/train/lightning_logs
"""


def test_load_train_config_reads_repository_file() -> None:
    """The repository training configuration loads."""
    config = load_train_config(Path("config/train.yaml"))

    assert config.name == "tiny_stories"
    assert config.sequence_length == 64
    assert config.model.hidden_size == 128
    assert config.scheduler_type == "cosine_warmup"
    assert config.data_dir == Path("data/tokenized/tiny_stories")
    assert config.enable_progress_bar is True


def test_load_train_config_rejects_split_fractions_that_leave_no_train(
    tmp_path: Path,
) -> None:
    """A split that consumes the whole set is rejected."""
    path = tmp_path / "train.yaml"
    path.write_text(
        _VALID.replace("test_size: 0.2", "test_size: 0.6").replace(
            "val_size: 0.15",
            "val_size: 0.5",
        ),
        encoding="utf-8",
    )

    with pytest.raises(InvalidTrainConfigError, match="less than 1"):
        load_train_config(path)


def test_load_train_config_rejects_persistent_workers_without_workers(
    tmp_path: Path,
) -> None:
    """Persistent workers require a worker process."""
    path = tmp_path / "train.yaml"
    path.write_text(
        _VALID.replace("persistent_workers: false", "persistent_workers: true"),
        encoding="utf-8",
    )

    with pytest.raises(InvalidTrainConfigError, match="persistent_workers"):
        load_train_config(path)


def test_load_train_config_rejects_a_missing_progress_bar_flag(
    tmp_path: Path,
) -> None:
    """A training file without enable_progress_bar is rejected."""
    path = tmp_path / "train.yaml"
    path.write_text(
        _VALID.replace("enable_progress_bar: false\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(InvalidTrainConfigError, match="enable_progress_bar"):
        load_train_config(path)


def test_load_train_config_rejects_a_missing_file(tmp_path: Path) -> None:
    """A missing configuration file is rejected."""
    with pytest.raises(InvalidTrainConfigError, match="does not exist"):
        load_train_config(tmp_path / "missing.yaml")
