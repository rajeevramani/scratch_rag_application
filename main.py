import asyncio
from scratch_rag_application.utils.logging_config import setup_logging
from scratch_rag_application.utils.cli_handler import parse_arguments
import logging
from scratch_rag_application.loader.url_loader import URLLoader
from scratch_rag_application.text_splitter.splitter_factory import TextSplitterFactory
from scratch_rag_application.embedding.embedding_factory import EmbeddingFactory
from scratch_rag_application.config.config_handler import ConfigHandler
from scratch_rag_application.vector_store.vector_store_factory import VectorStoreFactory
from scratch_rag_application.visualization.visualization_factory import VisualizationFactory
from typing import Optional
from scratch_rag_application.utils.chroma_inspector import inspect_chroma

setup_logging()
logger = logging.getLogger(__name__)


async def load_data(config: ConfigHandler) -> None:
    """Load and process documents into the vector store."""
    # Load documents
    url_loader = URLLoader()
    docs = await url_loader.load_urls()
    logger.info(f"Documents retrieved: {len(docs)}")

    # Split documents
    t_factory = TextSplitterFactory(config)
    splitter = t_factory.create_splitter()
    split_docs = splitter.split_documents(docs)
    logger.info(f"Documents split into {len(split_docs)} chunks")

    # Create embeddings
    embedding_factory = EmbeddingFactory(config)
    embedder = embedding_factory.create_embedder()

    # Initialize vector store
    vector_factory = VectorStoreFactory(config)
    vector_store = vector_factory.create_store(embedder)

    # Clear existing data if reload flag is set
    vector_store.delete()

    # Add documents to vector store
    if vector_store.add_documents(split_docs):
        logger.info("Successfully added documents to vector store")
    else:
        logger.error("Failed to add documents to vector store")

    return vector_store


async def query_store(query: str, config: ConfigHandler, viz_type: Optional[str] = None) -> None:
    """
    Execute a query against the existing vector store and show sorted results by relevance.

    Args:
        query: Search query string
        config: Configuration handler instance
        viz_type: Type of visualization to generate
    """
    # Initialize embeddings and vector store
    embedding_factory = EmbeddingFactory(config)
    embedder = embedding_factory.create_embedder()
    vector_factory = VectorStoreFactory(config)
    vector_store = vector_factory.create_store(embedder)

    # Execute search with scores and sort by relevance (ascending distance scores)
    results = vector_store._store.similarity_search_with_score(query, k=4)
    sorted_results = sorted(results, key=lambda x: x[1])

    logger.info(f"Found {len(sorted_results)
                         } relevant documents for query: {query}")
    logger.info("Results sorted by relevance (lower score = more relevant):\n")

    for idx, (doc, score) in enumerate(sorted_results, 1):
        logger.info(f"Result {idx} (Similarity Score: {score:.4f}):")
        logger.info(f"Content: {doc.page_content[:250]}...")
        logger.info(f"Source: {doc.metadata}")

    # Handle visualization based on type
    if viz_type:
        viz_mapping = {
            'relevance_score': 'matplotlib',
            'document_comparison': 'document_comparison'
        }
        viz_type_internal = viz_mapping.get(viz_type)
        if viz_type_internal:
            visualizer = VisualizationFactory.create_visualizer(
                viz_type_internal)
            if viz_type == 'document_comparison':
                visualizer.visualize(query, sorted_results)
            else:
                visualizer.visualize(sorted_results)
            logger.info(f"Generated {viz_type} visualization")


async def main():
    # Load configuration
    config = ConfigHandler("config.yaml")

    # Parse command line arguments
    reload_data_flag, query, visualize, inspect = parse_arguments()

    try:
        if reload_data_flag:
            await load_data(config)

        if query:
            await query_store(query, config, visualize)

        if inspect:

            inspect_chroma()
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
