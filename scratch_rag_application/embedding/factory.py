# embeddings/factory.py
from typing import Dict, Type
from .base import BaseEmbedding
from .huggingface import HuggingFaceEmbedding
from .fastembed import FastEmbedEmbedding


class EmbeddingFactory:
    _embedders: Dict[str, Type[BaseEmbedding]] = {
        "huggingface": HuggingFaceEmbedding,
        "fastembed": FastEmbedEmbedding
    }

    def __init__(self, config: dict):
        self.config = config

    def create_embedder(self) -> BaseEmbedding:
        embedder_type = self.config.get("embeddings.type")
        embedder_class = self._embedders.get(embedder_type)

        if not embedder_class:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")

        return embedder_class(self.config)
