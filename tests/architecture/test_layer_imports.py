"""Import direction of the download layers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _layer_directory(relative: str) -> Path:
    """Return one source layer under the repository root.

    Args:
        relative: Path relative to the repository root.

    Returns:
        The absolute layer directory.

    Raises:
        AssertionError: The directory does not exist.
    """
    directory = _REPOSITORY_ROOT / relative
    if not directory.is_dir():
        message = f"Layer directory does not exist: {directory}"
        raise AssertionError(message)
    return directory


def _python_files(root: Path) -> list[Path]:
    """Return Python files under a layer, or fail when the scan is empty.

    Args:
        root: Directory that must contain the layer's Python files.

    Returns:
        The Python files found under ``root``.

    Raises:
        AssertionError: The directory is missing or contains no Python files.
    """
    if not root.is_dir():
        message = f"Layer directory does not exist: {root}"
        raise AssertionError(message)
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        message = f"Layer directory has no Python files: {root}"
        raise AssertionError(message)
    return files


def _forbidden_imports(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Return imports under ``root`` that start with a forbidden prefix.

    Args:
        root: Layer directory to scan.
        forbidden: Module prefixes that the layer must not import.

    Returns:
        ``path:module`` entries for each forbidden import.
    """
    return [
        f"{path}:{module}"
        for path in _python_files(root)
        for module in _imported_modules(path)
        if module.startswith(forbidden)
    ]


def test_domain_does_not_import_outer_layers() -> None:
    """Domain code stays free of application, infrastructure, and entrypoints."""
    offenders = _forbidden_imports(
        _layer_directory("src/tiny_story/domain"),
        (
            "tiny_story.application",
            "tiny_story.infrastructure",
            "tiny_story.entrypoints",
        ),
    )

    assert offenders == []


def test_application_does_not_import_infrastructure_or_entrypoints() -> None:
    """Application code depends on ports and domain types."""
    offenders = _forbidden_imports(
        _layer_directory("src/tiny_story/application"),
        ("tiny_story.infrastructure", "tiny_story.entrypoints"),
    )

    assert offenders == []


def test_domain_and_application_do_not_import_providers() -> None:
    """Dataset acquisition stays free of provider libraries until infrastructure."""
    prefixes = ("huggingface_hub", "http", "urllib", "yaml")
    roots = (
        _layer_directory("src/tiny_story/domain"),
        _layer_directory("src/tiny_story/application"),
    )
    offenders = [
        f"{path}:{module}"
        for root in roots
        for path in _python_files(root)
        for module in _imported_modules(path)
        if module == "yaml" or module.startswith(prefixes)
    ]

    assert offenders == []


def test_entrypoints_import_composition_rather_than_inner_layers() -> None:
    """Entrypoints reach the download through composition."""
    offenders = _forbidden_imports(
        _layer_directory("src/tiny_story/entrypoints"),
        (
            "tiny_story.application",
            "tiny_story.domain",
            "tiny_story.infrastructure",
        ),
    )

    assert offenders == []


def test_missing_layer_directory_fails(tmp_path: Path) -> None:
    """A missing layer is a failed scan, not an empty pass."""
    with pytest.raises(AssertionError, match="does not exist"):
        _python_files(tmp_path / "missing-domain")


def test_empty_layer_directory_fails(tmp_path: Path) -> None:
    """A layer directory with no Python files does not pass."""
    empty = tmp_path / "domain"
    empty.mkdir()

    with pytest.raises(AssertionError, match="no Python files"):
        _python_files(empty)


def test_forbidden_import_is_reported(tmp_path: Path) -> None:
    """A real forbidden import is reported instead of an empty offender list."""
    layer = tmp_path / "domain"
    layer.mkdir()
    source = layer / "leak.py"
    source.write_text("import tiny_story.infrastructure.datasets\n", encoding="utf-8")

    offenders = _forbidden_imports(layer, ("tiny_story.infrastructure",))

    assert offenders == [f"{source}:tiny_story.infrastructure.datasets"]
