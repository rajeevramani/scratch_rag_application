# splitters/base.py
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the text splitter with configuration parameters.

        Args:
            params (Dict[str, Any]): Configuration parameters for the splitter
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self._splitter = self._create_splitter()

    @abstractmethod
    def _create_splitter(self):
        """Create and return the specific text splitter instance."""
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the documents using the configured splitter.

        Args:
            documents (List[Document]): Documents to split

        Returns:
            List[Document]: Split documents
        """
        if not self._splitter:
            self.logger.error("Splitter not initialized")
            return documents

        try:
            return self._splitter.split_documents(documents)
        except Exception as e:
            self.logger.error(f"Error splitting documents: {str(e)}")
            return documents
