# factory.py
from typing import Dict, Type
from .base_splitter import BaseTextSplitter
from .markdown_splitter import MarkdownDocumentSplitter
from .recursive_splitter import RecursiveDocumentSplitter
from .sentence_transformer_splitter import SentenceTransformerDocumentSplitter
from .spacy_splitter import SpacyDocumentSplitter
# from .huggingface_splitter import HuggingfaceSplitter


class TextSplitterFactory:
    _splitters: Dict[str, Type[BaseTextSplitter]] = {
        "markdown": MarkdownDocumentSplitter,
        "recursive_character": RecursiveDocumentSplitter,
        "sentence_transformer": SentenceTransformerDocumentSplitter,
        "spacy": SpacyDocumentSplitter,
        # "huggingface": HuggingfaceSplitter
    }

    def __init__(self, config: dict):
        self.config = config

    def create_splitter(self) -> BaseTextSplitter:
        splitter_type = self.config.get("text_splitter.type")
        splitter_class = self._splitters.get(splitter_type)

        if not splitter_class:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")

        return splitter_class(self.config)
