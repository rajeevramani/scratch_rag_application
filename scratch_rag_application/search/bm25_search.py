# scoring/bm25_search.py
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from .base_search import BaseSearch


class BM25Search(BaseSearch):
    """BM25 search implementation."""

    def _initialize_search(self) -> None:
        """Initialize BM25 with empty index."""
        self._scorer = None
        self.documents = []
        self.doc_mapping = {}  # Maps content to Document objects

    def initialize_documents(self, documents: List[Document]) -> bool:
        """
        Initialize or update BM25 with documents.

        Args:
            documents: List of Documents to index

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info(f"Initializing BM25 with {
                             len(documents)} documents")

            # Store documents and create mapping
            self.documents = [doc.page_content for doc in documents]
            self.doc_mapping = {doc.page_content: doc for doc in documents}

            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.documents]

            # Create BM25 instance
            self._scorer = BM25Okapi(tokenized_docs)

            self.logger.info("BM25 initialization successful")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {str(e)}")
            return False

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Search documents using BM25 scoring.

        Args:
            query: Search query string
            k: Optional number of results to return

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if not self._scorer:
            self.logger.error("BM25 not initialized")
            return []

        try:
            k = k or self.k
            self.logger.info(f"Executing BM25 search for query: '{query}'")

            # Tokenize query and get scores
            tokenized_query = query.lower().split()
            scores = self._scorer.get_scores(tokenized_query)

            # Create document-score pairs and sort
            doc_scores = list(zip(self.documents, scores))
            sorted_docs = sorted(
                doc_scores, key=lambda x: x[1], reverse=True)[:k]

            # Convert to Document objects with scores
            results = [(self.doc_mapping[doc], score)
                       for doc, score in sorted_docs]

            self._validate_and_log_results(results, query)
            return results

        except Exception as e:
            self.logger.error(f"Error in BM25 search: {str(e)}")
            return []
