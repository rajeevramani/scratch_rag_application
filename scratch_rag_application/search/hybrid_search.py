# scoring/hybrid_search.py
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from .base_search import BaseSearch
from .bm25_search import BM25Search
from .vector_search import VectorSearch


class HybridSearch(BaseSearch):
    """Hybrid search combining BM25 and vector similarity."""

    def _initialize_search(self) -> None:
        """Initialize both BM25 and vector search components."""
        if not self.store:
            raise ValueError("Vector store is required for hybrid search")

        # Get weights from config
        self.bm25_weight = self.params.get("scoring.hybrid.bm25_weight", 0.3)
        self.vector_weight = self.params.get(
            "scoring.hybrid.vector_weight", 0.7)

        # Initialize individual searchers
        self.bm25_searcher = BM25Search(self.params)
        self.vector_searcher = VectorSearch(self.params, self.store)

    def initialize_documents(self, documents: List[Document]) -> bool:
        """Initialize BM25 with documents."""
        return self.bm25_searcher.initialize_documents(documents)

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Execute hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query string
            k: Optional number of results to return

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        try:
            k = k or self.k
            self.logger.info(f"Executing hybrid search for query: '{query}'")

            # Get results from both searches
            vector_results = self.vector_searcher.search(query, k=k)
            bm25_results = self.bm25_searcher.search(query, k=k)

            # Combine results
            combined_results = self._combine_results(
                vector_results, bm25_results)

            self._validate_and_log_results(combined_results[:k], query)
            return combined_results[:k]

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            return []

    def _combine_results(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Combine and rerank results from both search methods."""
        try:
            # Create mappings of document content to scores
            vector_scores = {doc.page_content: (score, doc)
                             for doc, score in vector_results}
            bm25_scores = {doc.page_content: score
                           for doc, score in bm25_results}

            # Combine unique documents
            all_docs = set(vector_scores.keys()) | set(bm25_scores.keys())

            # Calculate combined scores
            combined_scores = []
            for doc_content in all_docs:
                vector_score = vector_scores.get(doc_content, (0, None))[0]
                bm25_score = bm25_scores.get(doc_content, 0)

                # Combine scores using weights
                combined_score = (
                    self.vector_weight * (1 - vector_score) +
                    self.bm25_weight * bm25_score
                )

                doc = vector_scores.get(doc_content, (0, None))[1]
                if doc:
                    combined_scores.append((doc, combined_score))

            # Sort by combined score (descending)
            return sorted(combined_scores, key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"Error combining results: {str(e)}")
            return vector_results  # Fallback to vector results
