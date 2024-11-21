

from .base_splitter import BaseTextSplitter
from langchain_text_splitters import SpacyTextSplitter


class SpacyDocumentSplitter(BaseTextSplitter):
    """Implementation of spaCy-based text splitter."""

    def _create_splitter(self):
        return SpacyTextSplitter(
            chunk_size=self.params.get("text_splitter.spacy.chunk_size", 1000),
            chunk_overlap=self.params.get(
                "text_splitter.spacy.chunk_overlap", 200)
        )
