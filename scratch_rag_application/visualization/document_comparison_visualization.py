from .base_visualization import BaseVisualization
import matplotlib.pyplot as plt
from typing import List, Tuple, Set
from langchain_core.documents import Document
import os
from datetime import datetime
import re


class DocumentComparisonVisualization(BaseVisualization):
    """Visualization for comparing query with matched documents."""

    def _get_matching_terms(self, query: str, doc_text: str) -> Set[str]:
        """Find exact matching terms between query and document."""
        # Convert to lowercase and split into words
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', doc_text.lower()))
        return query_words.intersection(doc_words)

    def _create_highlighted_text(self, text: str, matching_terms: Set[str]) -> str:
        """Create text with matching terms highlighted."""
        highlighted = text
        for term in matching_terms:
            # Case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f'**{term}**', highlighted)
        return highlighted

    def visualize(self, query: str, results: List[Tuple[Document, float]]) -> None:
        """
        Create a document comparison visualization.

        Args:
            query: The search query
            results: List of (document, score) tuples from similarity search
        """
        try:
            # Setup the plot
            fig, ax = plt.subplots(figsize=(15, 3 + len(results) * 2))
            ax.axis('off')

            # Start with query at the top
            text_content = f"Query: {query}\n\n"

            # Add each result with highlighting
            for idx, (doc, score) in enumerate(results, 1):
                # Find matching terms
                matching_terms = self._get_matching_terms(
                    query, doc.page_content)

                # Create highlighted content
                highlighted_content = self._create_highlighted_text(
                    # Limit content length
                    doc.page_content[:300], matching_terms)

                # Add metadata and content
                text_content += f"Result {idx} (Score: {(1-score)*100:.1f}%)\n"
                text_content += f"Source: {
                    doc.metadata.get('source', 'N/A')}\n"
                text_content += f"Content: {highlighted_content}...\n\n"

            # Display text with basic Markdown-style formatting
            ax.text(0.05, 0.95, text_content,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    fontfamily='monospace',
                    fontsize=9,
                    wrap=True)

            # Adjust layout
            plt.tight_layout()

            # Save visualization
            viz_dir = 'visualizations'
            os.makedirs(viz_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(viz_dir, f'doc_comparison_{timestamp}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(
                f"Document comparison visualization saved to: {filename}")

        except Exception as e:
            self.logger.error(
                f"Error creating document comparison visualization: {str(e)}")
            if plt.get_fignums():
                plt.close()
