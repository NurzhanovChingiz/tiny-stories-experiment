.PHONY: run check delete zip kernel kernel-down kernel-terminal test map

run: check

check:
	uv run ruff format .
	uv run ruff check --fix src
	uv run mypy src
	uv run pytest --cov
	uv run xenon --max-absolute B --max-average A --max-modules B src/

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

kernel:
	@if [ ! -f docker/kernel/.env ]; then cp docker/kernel/.env.example docker/kernel/.env; fi
	@sed -i "s/^UID=.*/UID=$$(id -u)/; s/^GID=.*/GID=$$(id -g)/" docker/kernel/.env
	docker compose -f docker/kernel/docker-compose.yaml up -d --build kernel

kernel-down:
	docker compose -f docker/kernel/docker-compose.yaml down

kernel-terminal:
	docker compose -f docker/kernel/docker-compose.yaml exec kernel bash
test:
	uv run pytest --cov
map:
	uv run .cursor/skills/map-architecture/generate_architecture_map.py --output docs/architecture_map.md
