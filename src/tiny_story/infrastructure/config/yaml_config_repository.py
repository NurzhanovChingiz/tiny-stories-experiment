"""YAML configuration repository adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import yaml

from tiny_story.application.port.config_repository import ConfigRepository, T

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from typing import Any


class YamlConfigRepository(ConfigRepository):
    """Loads typed configuration from YAML files."""

    def __init__(
        self,
        config_paths: Mapping[type[Any], Path],
        parsers: Mapping[type[Any], Callable[[dict[str, Any]], object]],
    ) -> None:
        """Store the YAML file and parser for each configuration type.

        Args:
            config_paths: YAML file selected by configuration type.
            parsers: Callable that builds one configuration from a YAML mapping.

        Returns:
            None.
        """
        self._config_paths = config_paths
        self._parsers = parsers

    def get(self, config_type: type[T]) -> T:
        """Load one typed configuration from its YAML file.

        Args:
            config_type: Type that selects the YAML file and parser.

        Returns:
            The parsed configuration.

        Flow:
            1. Resolve the YAML path for the requested type.
            2. Load the YAML document.
            3. Require a mapping document.
            4. Parse the mapping into the requested type.
        """
        # 1. Resolve the YAML path for the requested type.
        path = self._config_paths[config_type]

        # 2. Load the YAML document.
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        # 3. Require a mapping document.
        if not isinstance(raw, dict):
            message = "Expected YAML mapping"
            raise TypeError(message)

        # 4. Parse the mapping into the requested type.
        parser = self._parsers[config_type]
        return cast("T", parser(raw))
