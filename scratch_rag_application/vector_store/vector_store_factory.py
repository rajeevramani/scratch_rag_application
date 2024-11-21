# vectorstore/factory.py
from typing import Dict, Type
from .base_vector_store import BaseVectorStore
from .chroma import ChromaVectorStore
from langchain_core.embeddings import Embeddings


class VectorStoreFactory:
    _stores: Dict[str, Type[BaseVectorStore]] = {
        "chroma": ChromaVectorStore
    }

    def __init__(self, config: dict):
        self.config = config

    def create_store(self, embedding: Embeddings) -> BaseVectorStore:
        store_type = self.config.get("vectorstore.type", "chroma")
        store_class = self._stores.get(store_type)

        if not store_class:
            raise ValueError(f"Unsupported vector store type: {store_type}")

        return store_class(self.config, embedding)
