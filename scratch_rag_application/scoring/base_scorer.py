# scoring/base_scorer.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging


class BaseScorer(ABC):
    """Abstract base class for document scoring."""

    def __init__(self, params: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self._scorer = self._create_scorer()

    @abstractmethod
    def _create_scorer(self):
        """Create and return the specific scorer instance."""
        pass

    @abstractmethod
    def score_documents(self, query: str, documents: List[str], k: int = 4) -> List[tuple[str, float]]:
        """Score documents against query and return top k results with scores."""
        pass
