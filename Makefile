.PHONY: run check delete zip download-training-files train

run: check

check:
	uv run ruff format .
	uv run ruff check --fix src tests
	uv run mypy src tests
	uv run pytest -cov
	uv run xenon --max-absolute B --max-average A --max-modules B src/ tests/
del:
		find . -type d \( \
	  -name '__pycache__' -o \
	  -name '.pytest_cache' -o \
	  -name '.ruff_cache' -o \
	  -name '.mypy_cache' -o \
	  -name '.hypothesis_cache' \
	\) -prune -exec rm -rf -- {} +

	find . -type f -name '.DS_Store' -delete
zip:
	zip -r cursor.zip .cursor

download-training-files:
	bash docker/training/download_bigfiles.sh

train:
	@if [ ! -f docker/training/.env ]; then cp docker/training/.env.example docker/training/.env; fi
	@sed -i "s/^UID=.*/UID=$$(id -u)/; s/^GID=.*/GID=$$(id -g)/" docker/training/.env
	docker compose -f docker/training/docker-compose.yaml up -d --build train
