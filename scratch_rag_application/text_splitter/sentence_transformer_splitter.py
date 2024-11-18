# splitters/sentence_transformer_splitter.py
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .base import BaseTextSplitter


class SentenceTransformerDocumentSplitter(BaseTextSplitter):

    def _create_splitter(self):
        return SentenceTransformersTokenTextSplitter(
            chunk_size=self.params.get(
                "text_splitter.sentence_transformer.chunk_size", 1000),
            chunk_overlap=self.params.get(
                "text_splitter.sentence_transformer.chunk_overlap", 200),

            model_name=self.params.get(
                "text_splitter.sentence_transformer.model_name", "all-MiniLM-L6-v2")
        )
