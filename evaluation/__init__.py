# evaluation/__init__.py
"""
Evaluation module for cross-species vocalization model.

Contains:
- evaluate.py: Full test set evaluation with comprehensive metrics
- confusion_matrix.py: Confusion matrix visualization and error analysis
- benchmark.py: Compare against baseline models (random, majority, simple CNN)
"""

from evaluation.evaluate import ModelEvaluator
from evaluation.confusion_matrix import ConfusionMatrixVisualizer
from evaluation.benchmark import BenchmarkRunner

__all__ = [
    "ModelEvaluator",
    "ConfusionMatrixVisualizer",
    "BenchmarkRunner",
]