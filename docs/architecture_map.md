# Architecture map

Auto-generated path-only tree. Do not edit by hand.

Root: `tiny-stories-experiment/`

```
tiny-stories-experiment/
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
│   │   └── gpu_pytorch.ipynb
│   └── check.ipynb
├── src/
│   └── tiny_story/
│       ├── application/
│       │   ├── port/
│       │   │   ├── __init__.py
│       │   │   ├── config_repository.py
│       │   │   ├── file_reader.py
│       │   │   ├── prediction_writer.py
│       │   │   └── run_logger.py
│       │   └── __init__.py
│       ├── domain/
│       │   ├── dataset/
│       │   │   ├── __init__.py
│       │   │   └── dataset_spec.py
│       │   ├── __init__.py
│       │   └── prediction.py
│       ├── infrastructure/
│       │   ├── config/
│       │   │   ├── __init__.py
│       │   │   └── yaml_config_repository.py
│       │   ├── logging/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_run_logger.py
│       │   ├── readers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_file_reader.py
│       │   ├── writers/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_prediction_writer.py
│       │   └── __init__.py
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_filesystem_file_reader.py
│   ├── test_prediction.py
│   └── test_yaml_config_repository.py
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
