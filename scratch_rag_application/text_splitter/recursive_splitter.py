# splitters/recursive_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseTextSplitter


class RecursiveDocumentSplitter(BaseTextSplitter):
    """Implementation of recursive character text splitter."""

    def _create_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.params.get(
                "text_splitter.recursive_character.chunk_size", 1000),
            chunk_overlap=self.params.get(
                "text_splitter.recursive_character.chunk_overlap", 200),
            length_function=len,
            separators=self.params.get("text_splitter.recursive_character.separators", [
                                       "\n\n", "\n", " ", ""])
        )
