# scoring/hybrid_search.py
import logging
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document


class HybridSearch:
    """Combines vector and BM25 search results."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.bm25_weight = config.get("scoring.hybrid.bm25_weight", 0.3)
        self.vector_weight = config.get("scoring.hybrid.vector_weight", 0.7)
        self.k = config.get("scoring.parameters.k", 4)

    def combine_results(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[str, float]],
        store: Any
    ) -> List[Tuple[Document, float]]:
        """
        Combine and rerank results from vector and BM25 search.
        """
        try:
            self.logger.info(f"Combining results: Vector={
                             len(vector_results)}, BM25={len(bm25_results)}")

            # Create a map of document content to their vector scores
            vector_scores = {doc.page_content: (
                score, doc) for doc, score in vector_results}

            # Create a map of document content to their BM25 scores
            bm25_scores = {content: score for content, score in bm25_results}

            # Combine unique documents from both searches
            all_docs = set(vector_scores.keys()) | set(bm25_scores.keys())
            self.logger.info(f"Total unique documents found: {len(all_docs)}")

            combined_scores = []
            for doc_content in all_docs:
                # Get scores (default to 0 if not found)
                vector_score = vector_scores.get(doc_content, (0, None))[0]
                bm25_score = bm25_scores.get(doc_content, 0)

                # Combine scores using weights
                combined_score = (
                    # Convert distance to similarity
                    self.vector_weight * (1 - vector_score) +
                    self.bm25_weight * bm25_score
                )

                # Log individual scores
                self.logger.debug(
                    f"Doc Score Breakdown - Vector: {1-vector_score:.4f}, "
                    f"BM25: {bm25_score:.4f}, Combined: {combined_score:.4f}"
                )

                combined_scores.append((doc_content, combined_score))

            # Sort by combined score and get top k
            sorted_docs = sorted(
                combined_scores, key=lambda x: x[1], reverse=True)[:self.k]

            # Convert back to Document objects with scores
            final_results = []
            for doc_content, score in sorted_docs:
                # Find original Document object either from vector results or store
                doc_info = vector_scores.get(doc_content)
                if doc_info:
                    doc = doc_info[1]
                    source = "vector"
                else:
                    # If document came from BM25 results, look it up in the store
                    results = store.get()
                    try:
                        doc_idx = results['documents'].index(doc_content)
                        doc = Document(
                            page_content=doc_content,
                            metadata=results['metadatas'][doc_idx]
                        )
                        source = "bm25"
                    except (ValueError, KeyError):
                        continue

                if doc:
                    final_results.append((doc, score))
                    self.logger.info(
                        f"Result from {source} search - Score: {score:.4f}, "
                        f"Content: {doc.page_content[:100]}..."
                    )

            return final_results

        except Exception as e:
            self.logger.error(f"Error combining search results: {str(e)}")
            # Fallback to vector search results
            return vector_results[:self.k]
