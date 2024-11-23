# scoring/bm25_scorer.py
from rank_bm25 import BM25Okapi
from typing import List, Tuple
from .base_scorer import BaseScorer


class BM25Scorer(BaseScorer):
    """Implementation using BM25 scoring."""

    def _create_scorer(self) -> None:
        """BM25 will be initialized with documents during initialization."""
        return None

    def initialize(self, documents: List[str]):
        """Initialize BM25 with a set of documents."""
        try:
            self.logger.info(f"Initializing BM25 with {
                             len(documents)} documents")
            # Tokenize documents
            self.tokenized_docs = [doc.lower().split() for doc in documents]
            self.documents = documents
            # Create BM25 instance
            self._scorer = BM25Okapi(self.tokenized_docs)
            self.logger.info("BM25 initialization successful")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {str(e)}")
            return False

    def score_documents(self, query: str, documents: List[str], k: int = 4) -> List[Tuple[str, float]]:
        try:
            self.logger.info(f"Scoring query '{query}' against {
                             len(documents)} documents")
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get scores
            scores = self._scorer.get_scores(tokenized_query)

            # Create document-score pairs and sort
            doc_scores = list(zip(self.documents, scores))
            sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)

            # Log top scores
            for i, (doc, score) in enumerate(sorted_docs[:k], 1):
                self.logger.debug(f"BM25 Result {i}: Score={
                                  score:.4f}, Doc preview: {doc[:100]}...")

            return sorted_docs[:k]

        except Exception as e:
            self.logger.error(f"Error scoring documents: {str(e)}")
            return []
