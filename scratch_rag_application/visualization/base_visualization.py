# scratch_rag_application/visualization/base_visualization.py
from abc import ABC, abstractmethod
import logging
from typing import List, Tuple
from langchain_core.documents import Document


class BaseVisualization(ABC):
    """Abstract base class for visualizations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def visualize(self, results: List[Tuple[Document, float]]) -> None:
        """
        Create visualization for search results.

        Args:
            results: List of (document, score) tuples from similarity search
        """
        pass

    def _preprocess_results(self, results: List[Tuple[Document, float]]
                            ) -> Tuple[List[float], List[str]]:
        """
        Preprocess results for visualization.

        Args:
            results: List of (document, score) tuples

        Returns:
            Tuple containing lists of scores and labels
        """
        try:
            # Convert distance scores to similarity percentages
            scores = [(1 - score) * 100 for _, score in results]
            # Truncate document content for labels
            labels = [doc.page_content[:50] + "..." for doc, _ in results]
            return scores, labels
        except Exception as e:
            self.logger.error(f"Error preprocessing results: {str(e)}")
            return [], []
