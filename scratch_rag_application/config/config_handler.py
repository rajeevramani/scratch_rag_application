import yaml
from typing import Any, List, Union
from pathlib import Path
import logging
from functools import reduce
import os


class ConfigHandler:
    def __init__(self, yaml_file: Union[str, Path]):
        """
        Initialize ConfigHandler with a YAML file path.

        Args:
            yaml_file (Union[str, Path]): Path to the YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        yaml_path = Path(
            "/home/rajeevramani/personal/projects/llm/scratch_rag_application/config.yaml")
        self.logger.info(f"path: {yaml_path}")
        if not yaml_path.is_absolute():
            yaml_path = Path(os.getcwd()) / yaml_path

        # Resolve any relative path components (../../ etc)
        self.yaml_file = yaml_path.resolve()
        self.config = self._load_yaml()

    def _load_yaml(self) -> dict:
        """Load the YAML file into a dictionary."""
        if not self.yaml_file.exists():
            self.logger.error(f"Config file not found: {self.yaml_file}")
            raise FileNotFoundError(f"Config file not found: {self.yaml_file}")

        with open(self.yaml_file, 'r') as f:
            return yaml.safe_load(f)

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a value from the config using dot notation.

        Args:
            path (str): Path to the config value using dot notation
                       (e.g., 'pipeline.sources.website.urls')
            default (Any): Default value to return if path doesn't exist

        Returns:
            Any: The value at the specified path, or default if not found
        """
        try:
            return reduce(
                lambda d, key: d[key],
                path.split('.'),
                self.config
            )
        except (KeyError, TypeError):
            self.logger.error("KeyError or TypeError")
            return default

    def get_all_paths(self, prefix: str = '') -> List[str]:
        """
        Get all possible configuration paths in dot notation.

        Args:
            prefix (str): Optional prefix for the paths

        Returns:
            List[str]: List of all possible configuration paths
        """
        def _get_paths(d: dict, current_path: str) -> List[str]:
            paths = []
            for key, value in d.items():
                new_path = f"{current_path}.{key}" if current_path else key
                paths.append(new_path)
                if isinstance(value, dict):
                    paths.extend(_get_paths(value, new_path))
            return paths

        return _get_paths(self.config, prefix)
