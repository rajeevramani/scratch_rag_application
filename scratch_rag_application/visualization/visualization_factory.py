from typing import Dict, Type
from .base_visualization import BaseVisualization
from .matplotlib_visualization import MatplotlibVisualization
from .document_comparison_visualization import DocumentComparisonVisualization


class VisualizationFactory:
    """Factory for creating visualization instances."""

    _visualizers: Dict[str, Type[BaseVisualization]] = {
        "matplotlib": MatplotlibVisualization,
        "document_comparison": DocumentComparisonVisualization
    }

    @classmethod
    def create_visualizer(cls, visualizer_type: str = "matplotlib") -> BaseVisualization:
        """
        Create and return a visualizer instance.

        Args:
            visualizer_type: Type of visualizer to create

        Returns:
            BaseVisualization: Instance of the requested visualizer
        """
        visualizer_class = cls._visualizers.get(visualizer_type)

        if not visualizer_class:
            raise ValueError(f"Unsupported visualizer type: {visualizer_type}")

        return visualizer_class()
