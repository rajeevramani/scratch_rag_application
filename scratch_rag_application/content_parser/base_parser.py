# content_parser/base_parser.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import logging


@dataclass
class ParsedContent:
    """Container for parsed content section"""
    content: str
    content_type: str
    section_id: str


class BaseContentParser(ABC):
    """Abstract base class for content parsers"""

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize parser with configuration parameters.

        Args:
            params: Configuration parameters for the parser
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params

    @abstractmethod
    def parse(self, content: str) -> List[ParsedContent]:
        """
        Parse content into meaningful sections.

        Args:
            content: Raw content to parse

        Returns:
            List of ParsedContent objects
        """
        pass

    def _validate_content(self, content: str) -> bool:
        """
        Validate content is parseable.

        Args:
            content: Content to validate

        Returns:
            bool: True if content is valid
        """
        if not content or not isinstance(content, str):
            self.logger.error("Invalid content provided")
            return False
        return True
