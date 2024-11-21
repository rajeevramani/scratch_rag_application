# utils/cli_handler.py
import argparse
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def parse_arguments() -> Tuple[bool, Optional[str], Optional[str], bool]:
    """
    Parse command line arguments for the RAG application.

    Returns:
        Tuple[bool, Optional[str], Optional[str], bool]: 
            (reload_data flag, query string if provided, visualization type, inspect flag)
    """
    parser = argparse.ArgumentParser(
        description='RAG Application for document processing and querying',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--reload-data',
        action='store_true',
        help='Force reload of all data into the vector store'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Execute similarity search with the provided query'
    )

    parser.add_argument(
        '--visualize',
        choices=['relevance_score', 'document_comparison'],
        help='Type of visualization to generate:\n'
             'relevance_score: Show relevance scores as bar chart\n'
             'document_comparison: Show document matches with highlighted terms'
    )

    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect ChromaDB contents'
    )

    args = parser.parse_args()

    # If no arguments provided, show help and exit
    if not any(vars(args).values()):
        parser.print_help()
        return False, None, None, False

    return args.reload_data, args.query, args.visualize, args.inspect
