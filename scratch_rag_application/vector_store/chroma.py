# vectorstore/chroma.py
from langchain_chroma import Chroma
from .base_vector_store import BaseVectorStore
import os
from typing import Optional, List, Tuple
from langchain_core.documents import Document
from ..scoring.bm25_scorer import BM25Scorer
from ..scoring.hybrid_search import HybridSearch


class ChromaVectorStore(BaseVectorStore):
    """Implementation using Chroma vector store with hybrid search capability."""

    def __init__(self, params: dict, embedding):
        self.hybrid_search = HybridSearch(params)
        self.bm25_scorer = BM25Scorer(params)
        super().__init__(params, embedding)
        # Initialize BM25 with all documents during addition
        self._initialize_bm25()

    def _create_store(self) -> Chroma:
        """Create and return the Chroma vector store instance."""
        try:
            persist_directory = self.params.get(
                "vectorstore.chroma.persist_directory", "./chroma_db")
            collection_name = self.params.get(
                "vectorstore.chroma.collection_name", "default")

            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)

            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding,
                persist_directory=persist_directory
            )
        except Exception as e:
            self.logger.error(
                f"Error initializing Chroma vector store: {str(e)}")
            return None

    def _initialize_bm25(self):
        """Initialize BM25 with all documents in the store."""
        try:
            if self._store:
                results = self._store.get()
                if results and len(results['documents']) > 0:
                    self.all_documents = results['documents']
                    self.logger.info(
                        f"Retrieved {len(self.all_documents)} documents for BM25 initialization")
                    # Initialize BM25 with all documents
                    success = self.bm25_scorer.initialize(self.all_documents)
                    if success:
                        self.logger.info("BM25 initialization successful")
                    else:
                        self.logger.error("BM25 initialization failed")
        except Exception as e:
            self.logger.error(f"Error initializing BM25: {str(e)}")

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> bool:
        """Add documents to both vector store and BM25."""
        success = super().add_documents(documents, ids)
        if success:
            self.logger.info(
                "Documents added to vector store, reinitializing BM25")
            self._initialize_bm25()
        return success

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining vector similarity and BM25 scores.
        """
        try:
            # Get k from config if not provided
            k = k or self.params.get("scoring.parameters.k", 4)

            self.logger.info(f"Executing hybrid search for query: '{query}'")

            # Get vector search results
            vector_results = self._store.similarity_search_with_score(
                query,
                k=k
            )
            self.logger.info(f"Vector search found {
                             len(vector_results)} results")

            # Log vector search results
            for i, (doc, score) in enumerate(vector_results, 1):
                self.logger.debug(
                    f"Vector Result {i}: Score={score:.4f}, "
                    f"Content: {doc.page_content[:100]}..."
                )

            # Get BM25 results across ALL documents
            bm25_results = self.bm25_scorer.score_documents(
                query,
                self.all_documents,
                k=k
            )
            self.logger.info(f"BM25 search found {len(bm25_results)} results")

            # Combine results
            combined_results = self.hybrid_search.combine_results(
                vector_results,
                bm25_results,
                self._store
            )

            self.logger.info(f"Final combined results: {
                             len(combined_results)}")
            return combined_results

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            self.logger.info("Falling back to vector search only")
            return self._store.similarity_search_with_score(query, k=k)

    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """Delete the entire collection."""
        if not self._store:
            self.logger.error("Vector store not initialized")
            return False
        try:
            self._store.delete_collection()
            self._store = self._create_store()
            self.logger.info("Deleted collection and recreated store")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            return False
