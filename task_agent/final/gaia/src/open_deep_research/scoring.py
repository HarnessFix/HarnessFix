"""Scoring utilities for GAIA benchmark.

Directly re-exports normalize_str() and question_scorer() from the
upstream gaia_scorer.py (downloaded from smolagents/examples/open_deep_research).

DO NOT MODIFY THIS FILE — eval_gaia.py and run_gaia_entry.py depend on it.
Public API:
    normalize_str(input_str, remove_punct=True) -> str
    question_scorer(model_answer, ground_truth) -> bool
"""

from .scripts.gaia_scorer import normalize_str, question_scorer

__all__ = ["normalize_str", "question_scorer"]
