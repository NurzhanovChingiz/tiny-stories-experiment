"""YAML configuration for Lightning LSTM training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from tiny_story.domain.entities.language_model import CausalLmSpec
from tiny_story.domain.errors import InvalidTrainConfigError
from tiny_story.infrastructure.config.yaml_config_repository import YamlConfigRepository

_SCHEDULER_TYPES = frozenset({"cosine_warmup", "cosine", "step", "onecycle"})
_PRECISION_NAMES = frozenset(
    {
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        "64",
        "32",
        "16",
        "bf16",
    }
)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Train, validation, and test fractions for story sequences.

    Attributes:
        test_size: Fraction of sequences held out for test.
        val_size: Fraction of sequences held out for validation.
    """

    test_size: float
    val_size: float


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Validated settings for one Lightning training run.

    Attributes:
        name: Dataset name used for the run log file.
        seed: Integer seed for shuffling and Lightning.
        data_dir: Directory of tokenized JSONL files.
        vocab_file: Vocabulary file name or path resolved under ``data_dir``.
        sequence_length: Padded and truncated window length, including markers.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        scheduler_type: Lightning scheduler name.
        warmup_epochs: Warmup length for cosine-warmup schedules.
        eta_min: Minimum learning rate for cosine schedules.
        step_size: Epoch period for StepLR.
        gamma: StepLR multiplicative decay.
        max_lr: Peak learning rate for OneCycleLR.
        min_delta: Early-stopping improvement threshold.
        patience: Epochs without improvement before stopping.
        num_workers: DataLoader worker count.
        pin_memory: Whether DataLoader pins host memory.
        persistent_workers: Whether DataLoader workers stay alive.
        amp: Whether mixed precision from ``precision`` is used.
        precision: Lightning precision name.
        enable_progress_bar: Whether Lightning draws a progress bar.
        model: LSTM width settings.
        split: Held-out fractions.
        checkpoint_dir: Directory for the best checkpoint.
        log_dir: Directory for CSV Lightning logs.
    """

    name: str
    seed: int
    data_dir: Path
    vocab_file: Path
    sequence_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler_type: str
    warmup_epochs: int
    eta_min: float
    step_size: int
    gamma: float
    max_lr: float
    min_delta: float
    patience: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    amp: bool
    precision: str
    enable_progress_bar: bool
    model: CausalLmSpec
    split: SplitConfig
    checkpoint_dir: Path
    log_dir: Path


def parse_train_config(raw: object) -> TrainConfig:
    """Validate and convert a YAML mapping into training configuration.

    Args:
        raw: Parsed YAML document.

    Returns:
        Validated training settings.

    Raises:
        InvalidTrainConfigError: The document contains missing or invalid fields.

    Flow:
        1. Require the root mapping and nested model and split mappings.
        2. Validate identity, paths, and integer training sizes.
        3. Validate optimization, scheduler, and early-stopping numbers.
        4. Validate loader flags, progress bar, precision, model width, and split fractions.
        5. Return the configuration.
    """
    # 1. Require the root mapping and nested model and split mappings.
    config = _mapping(raw, "config")
    model = _mapping(config.get("model"), "model")
    split = _mapping(config.get("split"), "split")

    # 2. Validate identity, paths, and integer training sizes.
    name = _text(config.get("name"), "name")
    seed = _integer(config.get("seed"), "seed", minimum=0)
    data_dir = Path(_text(config.get("data_dir"), "data_dir"))
    vocab_file = Path(_text(config.get("vocab_file"), "vocab_file"))
    sequence_length = _integer(
        config.get("sequence_length"), "sequence_length", minimum=2
    )
    batch_size = _integer(config.get("batch_size"), "batch_size", minimum=1)
    epochs = _integer(config.get("epochs"), "epochs", minimum=1)
    num_workers = _integer(config.get("num_workers"), "num_workers", minimum=0)
    warmup_epochs = _integer(config.get("warmup_epochs"), "warmup_epochs", minimum=0)
    step_size = _integer(config.get("step_size"), "step_size", minimum=1)
    patience = _integer(config.get("patience"), "patience", minimum=1)
    embedding_dim = _integer(
        model.get("embedding_dim"), "model.embedding_dim", minimum=1
    )
    hidden_size = _integer(model.get("hidden_size"), "model.hidden_size", minimum=1)
    num_layers = _integer(model.get("num_layers"), "model.num_layers", minimum=1)

    # 3. Validate optimization, scheduler, and early-stopping numbers.
    learning_rate = _number(config.get("learning_rate"), "learning_rate", minimum=0.0)
    weight_decay = _number(config.get("weight_decay"), "weight_decay", minimum=0.0)
    eta_min = _number(config.get("eta_min"), "eta_min", minimum=0.0)
    gamma = _number(config.get("gamma"), "gamma", minimum=0.0)
    max_lr = _number(config.get("max_lr"), "max_lr", minimum=0.0)
    min_delta = _number(config.get("min_delta"), "min_delta", minimum=0.0)
    dropout = _number(model.get("dropout"), "model.dropout", minimum=0.0, maximum=1.0)
    test_size = _number(
        split.get("test_size"), "split.test_size", minimum=0.0, maximum=1.0
    )
    val_size = _number(
        split.get("val_size"), "split.val_size", minimum=0.0, maximum=1.0
    )
    if test_size + val_size >= 1.0:
        message = "split.test_size + split.val_size must be less than 1"
        raise InvalidTrainConfigError(message)
    scheduler_type = _text(config.get("scheduler_type"), "scheduler_type")
    if scheduler_type not in _SCHEDULER_TYPES:
        message = "scheduler_type must be one of cosine_warmup, cosine, step, onecycle"
        raise InvalidTrainConfigError(message)
    if scheduler_type == "cosine_warmup" and (
        warmup_epochs < 1 or warmup_epochs >= epochs
    ):
        message = "warmup_epochs must be between 1 and epochs - 1 for cosine_warmup"
        raise InvalidTrainConfigError(message)

    # 4. Validate loader flags, progress bar, precision, model width, and split fractions.
    pin_memory = _flag(config.get("pin_memory"), "pin_memory")
    persistent_workers = _flag(config.get("persistent_workers"), "persistent_workers")
    amp = _flag(config.get("amp"), "amp")
    enable_progress_bar = _flag(
        config.get("enable_progress_bar"),
        "enable_progress_bar",
    )
    if persistent_workers and num_workers == 0:
        message = "persistent_workers requires num_workers to be greater than 0"
        raise InvalidTrainConfigError(message)
    precision = _text(config.get("precision"), "precision")
    if precision not in _PRECISION_NAMES:
        message = f"precision must be a Lightning precision name, got {precision}"
        raise InvalidTrainConfigError(message)
    checkpoint_dir = Path(_text(config.get("checkpoint_dir"), "checkpoint_dir"))
    log_dir = Path(_text(config.get("log_dir"), "log_dir"))

    # 5. Return the configuration.
    return TrainConfig(
        name=name,
        seed=seed,
        data_dir=data_dir,
        vocab_file=vocab_file,
        sequence_length=sequence_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        step_size=step_size,
        gamma=gamma,
        max_lr=max_lr,
        min_delta=min_delta,
        patience=patience,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        amp=amp,
        precision=precision,
        enable_progress_bar=enable_progress_bar,
        model=CausalLmSpec(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ),
        split=SplitConfig(test_size=test_size, val_size=val_size),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )


def load_train_config(path: Path) -> TrainConfig:
    """Load and validate a training configuration file.

    Args:
        path: YAML training configuration.

    Returns:
        The parsed configuration.

    Raises:
        InvalidTrainConfigError: The file is missing, unreadable, or invalid.

    Flow:
        1. Reject a missing or unreadable configuration file.
        2. Load the mapping and report read, decode, YAML, and shape failures.
    """
    # 1. Reject a missing or unreadable configuration file.
    try:
        exists = path.is_file()
    except OSError as exc:
        message = f"Could not read train configuration: {path}"
        raise InvalidTrainConfigError(message) from exc
    if not exists:
        message = f"Train configuration file does not exist: {path}"
        raise InvalidTrainConfigError(message)

    # 2. Load the mapping and report read, decode, YAML, and shape failures.
    repository = YamlConfigRepository(
        {TrainConfig: path},
        {TrainConfig: parse_train_config},
    )
    try:
        return repository.get(TrainConfig)
    except (OSError, UnicodeError) as exc:
        message = f"Could not read train configuration: {path}"
        raise InvalidTrainConfigError(message) from exc
    except yaml.YAMLError as exc:
        message = f"Train configuration is not valid YAML: {path}"
        raise InvalidTrainConfigError(message) from exc
    except TypeError as exc:
        message = "Train configuration must be a YAML mapping"
        raise InvalidTrainConfigError(message) from exc


def _mapping(value: object, field: str) -> dict[str, object]:
    """Validate and return a string-keyed mapping.

    Args:
        value: Candidate configuration value.
        field: Field path used in validation errors.

    Returns:
        The validated mapping with string keys.

    Raises:
        InvalidTrainConfigError: The value is not a dictionary or has non-string keys.
    """
    if not isinstance(value, dict):
        message = f"{field} must be a mapping"
        raise InvalidTrainConfigError(message)
    if not all(isinstance(key, str) for key in value):
        message = f"{field} must use string keys"
        raise InvalidTrainConfigError(message)
    return cast("dict[str, object]", value)


def _text(value: object, field: str) -> str:
    """Return a non-empty string without surrounding whitespace.

    Args:
        value: Candidate configuration value.
        field: Field path used in validation errors.

    Returns:
        The validated string.

    Raises:
        InvalidTrainConfigError: The value is missing, empty, or padded.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{field} must be a non-empty string without surrounding whitespace"
        raise InvalidTrainConfigError(message)
    return value


def _flag(value: object, field: str) -> bool:
    """Return a boolean configuration flag.

    Args:
        value: Candidate flag value.
        field: Field name used in the error.

    Returns:
        The boolean value.

    Raises:
        InvalidTrainConfigError: The value is not a boolean.
    """
    if not isinstance(value, bool):
        message = f"{field} must be true or false"
        raise InvalidTrainConfigError(message)
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    """Return an integer at or above ``minimum``.

    Args:
        value: Candidate integer.
        field: Field path used in validation errors.
        minimum: Inclusive lower bound.

    Returns:
        The validated integer.

    Raises:
        InvalidTrainConfigError: The value is not an int or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{field} must be an integer >= {minimum}"
        raise InvalidTrainConfigError(message)
    return value


def _number(
    value: object,
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    """Return a finite real number inside the requested bounds.

    Args:
        value: Candidate number.
        field: Field path used in validation errors.
        minimum: Inclusive lower bound.
        maximum: Inclusive upper bound, or None when unbounded above.

    Returns:
        The validated number as float.

    Raises:
        InvalidTrainConfigError: The value is not a number or is out of range.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"{field} must be a number"
        raise InvalidTrainConfigError(message)
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        if maximum is None:
            message = f"{field} must be a number >= {minimum}"
        else:
            message = f"{field} must be a number between {minimum} and {maximum}"
        raise InvalidTrainConfigError(message)
    return number
