# utils/chroma_inspector.py
import logging
from typing import Optional, List
import chromadb

logger = logging.getLogger(__name__)


class ChromaInspector:
    """Utility class to inspect ChromaDB contents."""

    def __init__(self, persist_directory: str = "./chroma_db",
                 collection_name: str = "kong_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_collection_info(self) -> dict:
        """Get basic information about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

    def list_documents(self, limit: Optional[int] = None) -> List[dict]:
        """List all documents in the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.get()

            documents = []
            for idx, (doc_id, document, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
                if limit and idx >= limit:
                    break
                documents.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata
                })
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def search_documents(self, query_text: str, n_results: int = 5) -> List[dict]:
        """Search documents using raw ChromaDB query."""
        try:
            collection = self.client.get_collection(self.collection_name)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

            documents = []
            if results['documents']:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    documents.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance
                    })
            return documents
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []


def inspect_chroma():
    """Command line utility to inspect ChromaDB contents."""
    inspector = ChromaInspector()

    # Get collection info
    info = inspector.get_collection_info()
    logger.info("\nCollection Information:")
    logger.info("-" * 50)
    for key, value in info.items():
        logger.info(f"{key}: {value}")

    # List some documents
    logger.info("\nDocument Samples:")
    logger.info("-" * 50)
    documents = inspector.list_documents(limit=5)  # Show first 5 documents
    for idx, doc in enumerate(documents, 1):
        logger.info(f"\nDocument {idx}:")
        logger.info(f"ID: {doc['id']}")
        logger.info(f"Source: {doc['metadata'].get('source', 'N/A')}")
        logger.info(f"Content Preview: {doc['content'][:200]}...")
        logger.info("-" * 50)


if __name__ == "__main__":
    inspect_chroma()
