# content_parser/factory.py
from typing import Dict, Type
from .base_parser import BaseContentParser
from .qa_parser import QAContentParser


class ContentParserFactory:
    """Factory for creating content parser instances."""

    _parsers: Dict[str, Type[BaseContentParser]] = {
        "qa": QAContentParser
    }

    def __init__(self, config: dict):
        self.config = config

    def create_parser(self, parser_type: str) -> BaseContentParser:
        """
        Create and return a parser instance.

        Args:
            parser_type: Type of parser to create

        Returns:
            BaseContentParser: Instance of the requested parser
        """
        parser_class = self._parsers.get(parser_type)

        if not parser_class:
            raise ValueError(f"Unsupported parser type: {parser_type}")

        return parser_class(self.config)
