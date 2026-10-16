# Architecture map

Auto-generated path-only tree. Do not edit by hand.

Root: `tiny-stories-experiment/`

```
tiny-stories-experiment/
├── config/
│   └── dataset.yaml
├── docker/
│   └── kernel/
│       ├── docker-compose.yaml
│       ├── Dockerfile
│       └── download_bigfiles.sh
├── docs/
│   ├── architecture-map.meta.json
│   └── architecture_map.md
├── notebooks/
│   ├── 00_/
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
│       │   │   └── text_source.py
│       │   ├── results/
│       │   │   ├── __init__.py
│       │   │   └── download_outcome.py
│       │   ├── use_cases/
│       │   │   ├── __init__.py
│       │   │   └── download_dataset.py
│       │   └── __init__.py
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── __init__.py
│       │   │   ├── dataset_spec.py
│       │   │   ├── prediction.py
│       │   │   └── text.py
│       │   ├── __init__.py
│       │   └── errors.py
│       ├── entrypoints/
│       │   ├── cli/
│       │   │   ├── __init__.py
│       │   │   ├── __main__.py
│       │   │   └── download.py
│       │   └── __init__.py
│       ├── infrastructure/
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   ├── dataset_config.py
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
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_prediction_writer.py
│       │   └── __init__.py
│       ├── __init__.py
│       └── composition.py
├── tests/
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
│   │   └── test_acquire_dataset.py
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
│   │   │   └── test_download_command.py
│   │   └── __init__.py
│   ├── infrastructure/
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── test_dataset_config.py
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
