# scratch_rag_application/visualization/matplotlib_visualization.py
import matplotlib.pyplot as plt
from .base_visualization import BaseVisualization
from typing import List, Tuple
from langchain_core.documents import Document
import os
from datetime import datetime


class MatplotlibVisualization(BaseVisualization):
    """Implementation of visualization using matplotlib."""

    def visualize(self, results: List[Tuple[Document, float]]) -> None:
        """
        Create a bar chart visualization of search results using matplotlib.

        Args:
            results: List of (document, score) tuples from similarity search
        """
        try:
            # Preprocess the results
            scores, labels = self._preprocess_results(results)

            if not scores or not labels:
                self.logger.error("No valid data to visualize")
                return

            # Create figure with appropriate size
            plt.figure(figsize=(10, 6))

            # Create horizontal bar chart
            bars = plt.barh(range(len(scores)), scores)

            # Customize the plot
            plt.xlabel('Relevance Score (%)')
            plt.ylabel('Documents')
            plt.title('Search Results Relevance')
            plt.yticks(range(len(labels)), labels)

            # Add score labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                         f'{scores[i]:.1f}%',
                         ha='left', va='center', fontweight='bold')

            # Adjust layout
            plt.tight_layout()

            # Create visualizations directory if it doesn't exist
            viz_dir = 'visualizations'
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(viz_dir, f'search_results_{timestamp}.png')

            # Save the plot
            plt.savefig(filename)
            plt.close()

            self.logger.info(f"Visualization saved to: {filename}")

        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            if plt.get_fignums():
                plt.close()
