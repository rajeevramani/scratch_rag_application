# embeddings/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
import logging


class BaseEmbedding(ABC):
    """Abstract base class for embeddings using LangChain interface."""

    def __init__(self, params: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self._embedder = self._create_embedder()

    @abstractmethod
    def _create_embedder(self) -> Embeddings:
        """Create and return LangChain embeddings instance."""
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._embedder:
            self.logger.error("Embedder not initialized")
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            self.logger.error(f"Error embedding documents: {str(e)}")
            return []

    def embed_query(self, text: str) -> List[float]:
        if not self._embedder:
            self.logger.error("Embedder not initialized")
            return []
        try:
            return self._embedder.embed_query(text)
        except Exception as e:
            self.logger.error(f"Error embedding query: {str(e)}")
            return []
