# vectorstore/chroma.py
from langchain_chroma import Chroma
from .base_vector_store import BaseVectorStore
import os
from typing import Optional, List


class ChromaVectorStore(BaseVectorStore):
    """Implementation using Chroma vector store."""

    def _create_store(self) -> Chroma:
        try:
            persist_directory = self.params.get(
                "vectorstore.chroma.persist_directory", "./chroma_db")
            collection_name = self.params.get(
                "vectorstore.chroma.collection_name", "default")

            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)

            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding,
                persist_directory=persist_directory
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing Chroma vector store: {str(e)}")
            return None

    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """Delete the entire collection."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return False
        try:
            self._store.delete_collection()
            self._store = self._create_store()
            self.logger.info("Deleted collection and recreated store")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            return False
