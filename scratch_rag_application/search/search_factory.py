# scoring/search_factory.py
from typing import Dict, Type, Any
from .base_search import BaseSearch
from .bm25_search import BM25Search
from .vector_search import VectorSearch
from .hybrid_search import HybridSearch


class SearchFactory:
    """Factory for creating search implementations."""

    _searchers: Dict[str, Type[BaseSearch]] = {
        "bm25": BM25Search,
        "vector": VectorSearch,
        "hybrid": HybridSearch
    }

    def __init__(self, config: dict):
        self.config = config

    def create_searcher(self, store: Any = None) -> BaseSearch:
        """
        Create and return a search implementation.

        Args:
            store: Optional vector store instance for strategies that need it

        Returns:
            BaseSearch: Instance of the configured search implementation
        """
        search_type = self.config.get("scoring.type", "hybrid")
        search_class = self._searchers.get(search_type)

        if not search_class:
            raise ValueError(f"Unsupported search type: {search_type}")

        return search_class(self.config, store)
