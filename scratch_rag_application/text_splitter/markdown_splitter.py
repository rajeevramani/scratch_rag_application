# splitters/markdown_splitter.py
from langchain_text_splitters import MarkdownTextSplitter
from .base import BaseTextSplitter


class MarkdownDocumentSplitter(BaseTextSplitter):
    def create(self):
        params = self.config.get("markdown", {})
        return MarkdownTextSplitter(
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
            # Uses default separators if None
            separators=params.get("separators", None)
        )
