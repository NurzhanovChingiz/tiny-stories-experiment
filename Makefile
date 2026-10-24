.PHONY: run check delete zip kernel kernel-host kernel-down kernel-terminal kernel-logs kernel-smoke host-cuda test map download clean normalize tokenize

COMPOSE := docker compose -f docker/kernel/docker-compose.yaml

download:
	PYTHONPATH=src uv run python -m tiny_story.entrypoints.cli download

clean:
	PYTHONPATH=src uv run python -m tiny_story.entrypoints.cli clean

normalize:
	PYTHONPATH=src uv run python -m tiny_story.entrypoints.cli normalize

tokenize:
	PYTHONPATH=src uv run python -m tiny_story.entrypoints.cli tokenize

run: check

check:
	uv run ruff format .
	uv run ruff check --fix src tests
	uv run mypy src
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

kernel-host:
	sudo docker/kernel/setup_host_docker.sh

kernel:
	@if [ ! -f docker/kernel/.env ]; then cp docker/kernel/.env.example docker/kernel/.env; fi
	@sed -i "s/^UID=.*/UID=$$(id -u)/; s/^GID=.*/GID=$$(id -g)/" docker/kernel/.env
	mkdir -p artifacts/profiles
	@docker compose version >/dev/null 2>&1 || { \
	  echo "docker compose plugin is missing. Run: sudo make kernel-host"; \
	  exit 12; \
	}
	$(COMPOSE) up -d --build kernel

kernel-down:
	$(COMPOSE) down

kernel-terminal:
	$(COMPOSE) exec kernel bash

kernel-logs:
	$(COMPOSE) logs -f kernel

kernel-smoke:
	$(COMPOSE) exec kernel python /app/smoke.py

host-cuda:
	docker/kernel/setup_host_cuda.sh

test:
	uv run pytest --cov

map:
	uv run .cursor/skills/map-architecture/generate_architecture_map.py --output docs/architecture_map.md
