from langchain_huggingface import HuggingFaceEmbeddings
from .base import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):
    """Implementation using LangChain's HuggingFaceEmbeddings."""

    def _create_embedder(self) -> HuggingFaceEmbeddings:
        try:
            return HuggingFaceEmbeddings(
                model_name=self.params.get(
                    "embeddings.huggingface.model_name",
                    "sentence-transformers/all-mpnet-base-v2"
                ),
                model_kwargs=self.params.get(
                    "embeddings.huggingface.model_kwargs",
                    {"device": "cpu"}
                ),
                encode_kwargs=self.params.get(
                    "embeddings.huggingface.encode_kwargs",
                    {"normalize_embeddings": True}
                )
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing HuggingFace embeddings: {str(e)}")
            return None
