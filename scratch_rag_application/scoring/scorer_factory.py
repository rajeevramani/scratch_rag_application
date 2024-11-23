# scoring/scorer_factory.py
from typing import Dict, Type
from .base_scorer import BaseScorer
from .bm25_scorer import BM25Scorer


class ScorerFactory:
    """Factory for creating scorer instances."""

    _scorers: Dict[str, Type[BaseScorer]] = {
        "bm25": BM25Scorer
    }

    def __init__(self, config: dict):
        self.config = config

    def create_scorer(self) -> BaseScorer:
        scorer_type = self.config.get("scoring.type")
        scorer_class = self._scorers.get(scorer_type)

        if not scorer_class:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

        return scorer_class(self.config)
