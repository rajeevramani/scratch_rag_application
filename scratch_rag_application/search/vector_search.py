# scoring/vector_search.py
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from .base_search import BaseSearch


class VectorSearch(BaseSearch):
    """Pure vector-based search implementation."""

    def _initialize_search(self) -> None:
        """No additional initialization needed for vector search."""
        if not self.store:
            raise ValueError("Vector store is required for vector search")

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Search documents using vector similarity.

        Args:
            query: Search query string
            k: Optional number of results to return

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        try:
            k = k or self.k
            self.logger.info(f"Executing vector search for query: '{query}'")

            # Use the vector store's similarity search
            results = self.store.similarity_search_with_score(query, k=k)

            self._validate_and_log_results(results, query)
            return results

        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}")
            return []
