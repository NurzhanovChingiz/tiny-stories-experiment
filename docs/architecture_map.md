# Architecture map

Auto-generated path-only tree. Do not edit by hand.

Root: `tiny-stories-experiment/`

```
tiny-stories-experiment/
├── config/
│   ├── clean_text.yaml
│   ├── dataset.yaml
│   └── normalize_text.yaml
├── docker/
│   └── kernel/
│       ├── docker-compose.yaml
│       ├── Dockerfile
│       └── download_bigfiles.sh
├── docs/
│   ├── architecture-map.meta.json
│   └── architecture_map.md
├── notebooks/
│   ├── 00_environment/
│   │   └── gpu_pytorch.ipynb
│   └── check.ipynb
├── src/
│   └── tiny_story/
│       ├── application/
│       │   ├── ports/
│       │   │   ├── __init__.py
│       │   │   ├── config_repository.py
│       │   │   ├── dataset_source.py
│       │   │   ├── file_reader.py
│       │   │   ├── prediction_writer.py
│       │   │   ├── raw_dataset_store.py
│       │   │   ├── run_logger.py
│       │   │   ├── text_cleaner.py
│       │   │   ├── text_normalizer.py
│       │   │   └── text_source.py
│       │   ├── results/
│       │   │   ├── __init__.py
│       │   │   └── download_outcome.py
│       │   ├── use_cases/
│       │   │   ├── __init__.py
│       │   │   └── download_dataset.py
│       │   ├── __init__.py
│       │   └── existing_file_policy.py
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── __init__.py
│       │   │   ├── cleaning.py
│       │   │   ├── dataset_spec.py
│       │   │   ├── normalization.py
│       │   │   ├── prediction.py
│       │   │   └── text.py
│       │   ├── __init__.py
│       │   └── errors.py
│       ├── entrypoints/
│       │   ├── cli/
│       │   │   ├── __init__.py
│       │   │   ├── __main__.py
│       │   │   ├── app.py
│       │   │   ├── clean.py
│       │   │   ├── download.py
│       │   │   └── normalize.py
│       │   └── __init__.py
│       ├── infrastructure/
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   ├── clean_text_config.py
│       │   │   ├── dataset_config.py
│       │   │   ├── normalize_text_config.py
│       │   │   └── yaml_config_repository.py
│       │   ├── datasets/
│       │   │   ├── __init__.py
│       │   │   ├── filesystem_dataset_repository.py
│       │   │   └── huggingface_dataset_source.py
│       │   ├── io/
│       │   │   ├── __init__.py
│       │   │   └── read_text_jsonl.py
│       │   ├── logging/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_run_logger.py
│       │   ├── readers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_file_reader.py
│       │   ├── runs/
│       │   │   ├── __init__.py
│       │   │   ├── download_log_layout.py
│       │   │   └── inference_run_layout.py
│       │   ├── text_processing/
│       │   │   ├── clean_text/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── compiled_patterns.py
│       │   │   │   ├── composite_cleaner.py
│       │   │   │   ├── jsonl_story_cleaner.py
│       │   │   │   └── steps.py
│       │   │   ├── normalize_text/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── composite_normalizer.py
│       │   │   │   ├── jsonl_story_normalizer.py
│       │   │   │   └── steps.py
│       │   │   └── __init__.py
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_prediction_writer.py
│       │   └── __init__.py
│       ├── __init__.py
│       └── composition.py
├── tests/
│   ├── acceptance/
│   │   ├── features/
│   │   │   ├── cleaning_story_text.feature
│   │   │   ├── normalizing_story_text.feature
│   │   │   └── preparing_story_files.feature
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── story_text.py
│   │   ├── test_cleaning_story_text.py
│   │   ├── test_normalizing_story_text.py
│   │   └── test_preparing_story_files.py
│   ├── application/
│   │   ├── use_cases/
│   │   │   ├── __init__.py
│   │   │   └── test_download_dataset.py
│   │   └── __init__.py
│   ├── architecture/
│   │   ├── __init__.py
│   │   └── test_layer_imports.py
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── test_acquire_dataset.py
│   │   └── test_normalize_dataset.py
│   ├── domain/
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── test_dataset_spec.py
│   │   │   ├── test_prediction.py
│   │   │   └── test_text.py
│   │   └── __init__.py
│   ├── entrypoints/
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   ├── test_clean_command.py
│   │   │   ├── test_download_command.py
│   │   │   └── test_normalize_command.py
│   │   └── __init__.py
│   ├── infrastructure/
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── test_clean_text_config.py
│   │   │   ├── test_dataset_config.py
│   │   │   ├── test_normalize_text_config.py
│   │   │   └── test_yaml_config_repository.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── test_filesystem_dataset_repository.py
│   │   │   └── test_huggingface_dataset_source.py
│   │   ├── io/
│   │   │   ├── __init__.py
│   │   │   └── test_read_text_jsonl.py
│   │   ├── readers/
│   │   │   ├── __init__.py
│   │   │   └── test_filesystem_file_reader.py
│   │   ├── runs/
│   │   │   ├── __init__.py
│   │   │   ├── test_download_log_layout.py
│   │   │   └── test_inference_run_layout.py
│   │   ├── text_processing/
│   │   │   ├── clean_text/
│   │   │   │   ├── __init__.py
│   │   │   │   └── test_steps.py
│   │   │   ├── normalize_text/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── test_jsonl_story_normalizer.py
│   │   │   │   └── test_steps.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── __init__.py
├── .cursorignore
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```
