.PHONY: run check delete zip download-training-files download-kernel-files download prepare train-tokenizer train kernel test

run: check

check:
	uv run ruff format .
	uv run ruff check --fix src tests
	uv run mypy src tests
	uv run pytest --cov
	uv run xenon --max-absolute B --max-average A --max-modules B src/ tests/

delete:
	find . -type d \( \
	  -name '__pycache__' -o \
	  -name '.pytest_cache' -o \
	  -name '.ruff_cache' -o \
	  -name '.mypy_cache' -o \
	  -name '.hypothesis_cache' -o \
	  -name '.ipynb_checkpoints' \
	\) -prune -exec rm -rf -- {} +
	find . -type f -name '.DS_Store' -delete

zip:
	zip -r cursor.zip .cursor

download-training-files download-kernel-files:
	bash docker/kernel/download_bigfiles.sh

download:
	uv run python -m tiny_stories_experiment.entrypoints.cli download

prepare:
	uv run python -m tiny_stories_experiment.entrypoints.cli prepare

train-tokenizer:
	uv run python -m tiny_stories_experiment.entrypoints.cli train-tokenizer

kernel:
	@if [ ! -f docker/kernel/.env ]; then cp docker/kernel/.env.example docker/kernel/.env; fi
	@sed -i "s/^UID=.*/UID=$$(id -u)/; s/^GID=.*/GID=$$(id -g)/" docker/kernel/.env
	docker compose -f docker/kernel/docker-compose.yaml up -d --build kernel

test:
	uv run pytest --cov
