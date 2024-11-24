# scoring/base_search.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
import logging


class BaseSearch(ABC):
    """Abstract base class for search implementations."""

    def __init__(self, params: Dict[str, Any], store: Any = None):
        """
        Initialize search with configuration parameters.

        Args:
            params: Configuration parameters for the search
            store: Optional vector store instance for strategies that need it
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self.store = store
        self.k = params.get("scoring.parameters.k", 4)
        self._initialize_search()

    @abstractmethod
    def _initialize_search(self) -> None:
        """Initialize specific search implementation."""
        pass

    @abstractmethod
    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            k: Optional number of results to return, defaults to config value

        Returns:
            List of tuples containing (document, score)
        """
        pass

    def _validate_and_log_results(
        self, results: List[Tuple[Document, float]], query: str
    ) -> None:
        """Log search results for debugging."""
        self.logger.info(f"Found {len(results)} results for query: {query}")
        for i, (doc, score) in enumerate(results, 1):
            self.logger.debug(
                f"Result {i}: Score={score:.4f}, "
                f"Content preview: {doc.page_content[:100]}..."
            )
