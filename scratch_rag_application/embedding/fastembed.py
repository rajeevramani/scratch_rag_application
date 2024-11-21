# embeddings/fastembed.py
from langchain_community.embeddings import FastEmbedEmbeddings
from .base_embedding import BaseEmbedding


class FastEmbedEmbedding(BaseEmbedding):
    """Implementation using LangChain's FastEmbedEmbeddings."""

    def _create_embedder(self) -> FastEmbedEmbeddings:
        try:
            return FastEmbedEmbeddings(
                model_name=self.params.get(
                    "embeddings.fastembed.model_name",
                    "BAAI/bge-small-en-v1.5"
                ),
                max_length=self.params.get(
                    "embeddings.fastembed.max_length",
                    512
                ),
                batch_size=self.params.get(
                    "embeddings.fastembed.batch_size",
                    256
                )
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing FastEmbed embeddings: {str(e)}")
            return None
