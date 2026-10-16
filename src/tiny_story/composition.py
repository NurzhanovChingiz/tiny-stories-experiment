"""Wire dataset acquisition, cleaning, and text normalization from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tiny_story.application.use_cases.download_dataset import download_dataset
from tiny_story.domain.errors import CleanTextError, NormalizeTextError
from tiny_story.infrastructure.config.clean_text_config import load_clean_text_config
from tiny_story.infrastructure.config.dataset_config import (
    DatasetConfig,
    HuggingFaceSourceConfig,
    load_dataset_config,
)
from tiny_story.infrastructure.config.normalize_text_config import (
    load_normalize_text_config,
)
from tiny_story.infrastructure.datasets.filesystem_dataset_repository import (
    FilesystemDatasetRepository,
)
from tiny_story.infrastructure.datasets.huggingface_dataset_source import (
    HuggingFaceDatasetSource,
)
from tiny_story.infrastructure.logging.filesystem_run_logger import FilesystemRunLogger
from tiny_story.infrastructure.runs.download_log_layout import (
    clean_log_path,
    download_log_path,
    normalize_log_path,
)
from tiny_story.infrastructure.text_processing.clean_text.jsonl_story_cleaner import (
    JsonlStoryCleaner,
)
from tiny_story.infrastructure.text_processing.normalize_text.jsonl_story_normalizer import (
    JsonlStoryNormalizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tiny_story.application.results.download_outcome import DownloadOutcome


def acquire_dataset(
    config_path: Path,
    destination: Path | None = None,
    cache_dir: Path | None = None,
    log_path: Path | None = None,
) -> DownloadOutcome:
    """Acquire the dataset described by a configuration file.

    Args:
        config_path: YAML dataset configuration.
        destination: Raw directory that replaces ``dataset.destination`` when set.
        cache_dir: Hugging Face cache that replaces ``dataset.source.cache_dir`` when set.
        log_path: Run log file. Defaults to
            ``artifacts/download/logs/<dataset name>.log``.

    Returns:
        Which raw files were stored and which were already complete.

    Flow:
        1. Load the dataset configuration.
        2. Apply caller overrides to the destination and cache.
        3. Download the configured Hugging Face files into raw storage.
    """
    # 1. Load the dataset configuration.
    config = load_dataset_config(config_path)

    # 2. Apply caller overrides to the destination and cache.
    config = _with_overrides(config, destination=destination, cache_dir=cache_dir)

    # 3. Download the configured Hugging Face files into raw storage.
    run_logger = FilesystemRunLogger(
        _selected_log_path(config.spec.name, log_path, download_log_path)
    )
    try:
        run_logger.log(
            f"Loaded dataset configuration {config_path} for {config.spec.name} "
            f"into {config.destination} from {config.source.repository}"
        )
        source = HuggingFaceDatasetSource(
            cache_dir=config.source.cache_dir,
            repository=config.source.repository,
            revision=config.source.revision,
        )
        store = FilesystemDatasetRepository(config.destination)
        return download_dataset(
            config.spec,
            source,
            store,
            existing=config.existing,
            run_logger=run_logger,
        )
    finally:
        run_logger.close()


def _selected_log_path(
    dataset_name: str,
    log_path: Path | None,
    default_path: Callable[[Path, str], Path],
) -> Path:
    """Return the caller path or the stage default under artifacts.

    An explicit path is used as given. Otherwise ``default_path`` places the
    log at ``artifacts/<stage>/logs/<dataset name>.log``.

    Args:
        dataset_name: Dataset whose name selects the default log file.
        log_path: Caller-selected log file, or None for the stage default.
        default_path: Builds the stage log path from an artifacts root and a
            dataset name.

    Returns:
        The log file path.

    Flow:
        1. Use a caller-selected log file.
        2. Place the default log under the stage logs directory.
    """
    # 1. Use a caller-selected log file.
    if log_path is not None:
        return log_path

    # 2. Place the default log under the stage logs directory.
    return default_path(Path("artifacts"), dataset_name)


def _with_overrides(
    config: DatasetConfig,
    *,
    destination: Path | None,
    cache_dir: Path | None,
) -> DatasetConfig:
    """Return configuration with optional destination and cache overrides.

    Args:
        config: Parsed dataset configuration.
        destination: Replacement raw directory, or None to keep the configured one.
        cache_dir: Replacement Hugging Face cache, or None to keep the configured one.

    Returns:
        The configuration acquisition should use.
    """
    source = config.source
    if cache_dir is not None:
        source = HuggingFaceSourceConfig(
            repository=source.repository,
            revision=source.revision,
            cache_dir=cache_dir,
        )
    raw_dir = config.destination if destination is None else destination
    return DatasetConfig(
        spec=config.spec,
        destination=raw_dir,
        existing=config.existing,
        source=source,
    )


def clean_dataset(
    dataset_config_path: Path,
    clean_config_path: Path,
    output_dir: Path | None = None,
    log_path: Path | None = None,
) -> tuple[Path, ...]:
    """Clean each configured raw JSONL file into an interim file.

    Raw files are read and left unchanged. Interim files are written under
    ``data/interim/<dataset name>`` unless ``output_dir`` is set.

    Args:
        dataset_config_path: YAML file that names the raw dataset files.
        clean_config_path: YAML file that selects cleaning flags.
        output_dir: Directory that replaces the default interim directory.
        log_path: Run log file. Defaults to
            ``artifacts/clean/logs/<dataset name>.log``.

    Returns:
        The interim files that were written. Empty when saving cleaned text
        is disabled.

    Flow:
        1. Load the dataset and cleaning configuration and open the run log.
        2. Skip writing when cleaned text is not saved.
        3. Clean each raw JSONL file into the interim directory.
    """
    # 1. Load the dataset and cleaning configuration and open the run log.
    dataset = load_dataset_config(dataset_config_path)
    settings = load_clean_text_config(clean_config_path)
    run_logger = FilesystemRunLogger(
        _selected_log_path(dataset.spec.name, log_path, clean_log_path)
    )
    try:
        run_logger.log(
            f"Loaded cleaning configuration {clean_config_path} for {dataset.spec.name}"
        )

        # 2. Skip writing when cleaned text is not saved.
        if not settings.save_processed:
            run_logger.log(
                f"Skipping cleaned text for {dataset.spec.name} "
                "because save_processed is false"
            )
            return ()

        # 3. Clean each raw JSONL file into the interim directory.
        interim = (
            output_dir
            if output_dir is not None
            else Path("data/interim") / dataset.spec.name
        )
        cleaner = JsonlStoryCleaner(
            enabled=settings.enabled,
            page_header_pattern=settings.page_header_pattern,
            junk_line_pattern=settings.junk_line_pattern,
            reflow_hard_wrapped_lines=settings.reflow_hard_wrapped_lines,
            join_hyphenated_line_breaks=settings.join_hyphenated_line_breaks,
        )
        written: list[Path] = []
        for filename in dataset.spec.filenames:
            destination = interim / filename
            run_logger.log(f"Cleaning {filename}")
            try:
                cleaner.clean_file(dataset.destination / filename, destination)
            except CleanTextError as exc:
                run_logger.log(
                    f"Cleaning failed for {filename}: {exc}",
                    level="error",
                )
                raise
            run_logger.log(f"Wrote cleaned file {destination}")
            written.append(destination)
        run_logger.log(
            f"Cleaned {dataset.spec.name}: wrote {len(written)} file(s) in {interim}"
        )
        return tuple(written)
    finally:
        run_logger.close()


def normalize_dataset(
    dataset_config_path: Path,
    normalize_config_path: Path,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    log_path: Path | None = None,
) -> tuple[Path, ...]:
    """Normalize each cleaned JSONL file into a processed file.

    Cleaned files are read from ``data/interim/<dataset name>`` unless
    ``input_dir`` is set. Processed files are written under
    ``data/processed/<dataset name>`` unless ``output_dir`` is set.

    Args:
        dataset_config_path: YAML file that names the dataset files.
        normalize_config_path: YAML file that selects normalization flags.
        input_dir: Directory of cleaned JSONL files.
        output_dir: Directory that replaces the default processed directory.
        log_path: Run log file. Defaults to
            ``artifacts/normalize/logs/<dataset name>.log``.

    Returns:
        The processed files that were written. Empty when saving normalized
        text is disabled.

    Flow:
        1. Load the dataset and normalization configuration and open the run log.
        2. Skip writing when normalized text is not saved.
        3. Normalize each cleaned JSONL file into the processed directory.
    """
    # 1. Load the dataset and normalization configuration and open the run log.
    dataset = load_dataset_config(dataset_config_path)
    settings = load_normalize_text_config(normalize_config_path)
    run_logger = FilesystemRunLogger(
        _selected_log_path(dataset.spec.name, log_path, normalize_log_path)
    )
    try:
        run_logger.log(
            f"Loaded normalization configuration {normalize_config_path} "
            f"for {dataset.spec.name}"
        )

        # 2. Skip writing when normalized text is not saved.
        if not settings.save_processed:
            run_logger.log(
                f"Skipping normalized text for {dataset.spec.name} "
                "because save_processed is false"
            )
            return ()

        # 3. Normalize each cleaned JSONL file into the processed directory.
        cleaned = (
            input_dir
            if input_dir is not None
            else Path("data/interim") / dataset.spec.name
        )
        processed = (
            output_dir
            if output_dir is not None
            else Path("data/processed") / dataset.spec.name
        )
        normalizer = JsonlStoryNormalizer(
            enabled=settings.enabled,
            sentence_delimiter=settings.sentence_delimiter,
        )
        written: list[Path] = []
        for filename in dataset.spec.filenames:
            destination = processed / filename
            run_logger.log(f"Normalizing {filename}")
            try:
                normalizer.normalize_file(cleaned / filename, destination)
            except NormalizeTextError as exc:
                run_logger.log(
                    f"Normalization failed for {filename}: {exc}",
                    level="error",
                )
                raise
            run_logger.log(f"Wrote normalized file {destination}")
            written.append(destination)
        run_logger.log(
            f"Normalized {dataset.spec.name}: wrote {len(written)} file(s) "
            f"in {processed}"
        )
        return tuple(written)
    finally:
        run_logger.close()
