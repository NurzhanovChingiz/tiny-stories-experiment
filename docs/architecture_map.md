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
│   ├── data/
│   │   └── .gitkeep
│   └── check.ipynb
├── src/
│   └── tiny_story/
│       ├── application/
│       │   ├── port/
│       │   │   ├── __init__.py
│       │   │   └── run_logger.py
│       │   └── __init__.py
│       ├── infrastructure/
│       │   ├── logging/
│       │   │   ├── __init__.py
│       │   │   └── filesystem_run_logger.py
│       │   └── __init__.py
│       └── __init__.py
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
