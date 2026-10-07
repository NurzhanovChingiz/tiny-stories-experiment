"""Import-direction tests for the clean-architecture layers."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "tiny_stories_experiment"

FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": (
        "tiny_stories_experiment.application",
        "tiny_stories_experiment.infrastructure",
        "tiny_stories_experiment.entrypoints",
        "tiny_stories_experiment.composition",
        "huggingface_hub",
        "tokenizers",
        "typer",
    ),
    "application": (
        "tiny_stories_experiment.infrastructure",
        "tiny_stories_experiment.entrypoints",
        "tiny_stories_experiment.composition",
        "huggingface_hub",
        "tokenizers",
        "typer",
    ),
    "infrastructure": (
        "tiny_stories_experiment.entrypoints",
        "tiny_stories_experiment.composition",
    ),
}


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.level == 0
        ):
            modules.add(node.module)
    return modules


def _is_forbidden(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes
    )


def test_inner_layers_do_not_import_outer_layers() -> None:
    """Domain and application stay inward of infrastructure, composition, and the CLI."""
    violations = [
        f"{path.relative_to(PACKAGE)} imports {module}"
        for layer, prefixes in FORBIDDEN.items()
        for path in (PACKAGE / layer).rglob("*.py")
        for module in _imported_modules(path)
        if _is_forbidden(module, prefixes)
    ]
    assert violations == []
