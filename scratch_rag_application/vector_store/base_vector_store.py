# vectorstore/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logging


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, params: Dict[str, Any], embedding: Embeddings):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self.embedding = embedding
        self._store = self._create_store()

    @abstractmethod
    def _create_store(self):
        """Create and return the specific vector store instance."""
        pass

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> bool:
        """Add documents to the vector store."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return False
        try:
            # Generate UUIDs if no IDs provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(documents))]
            self._store.add_documents(documents=documents, ids=ids)
            return True
        except Exception as e:
            self.logger.error(
                f"Error adding documents to vector store: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 4,
                          filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Perform similarity search."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return []
        try:
            return self._store.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {str(e)}")
            return []

    def update_document(self, document_id: str, document: Document) -> bool:
        """Update a document in the store."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return False
        try:
            self._store.update_document(document_id, document)
            return True
        except Exception as e:
            self.logger.error(f"Error updating document: {str(e)}")
            return False

    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents from the store."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return False
        try:
            self._store.delete(ids=ids)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            return False
