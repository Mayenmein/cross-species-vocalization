# training/__init__.py
"""
Training module for cross-species vocalization model.

Contains:
- dataset.py: Data loading and preprocessing
- augmentations.py: Audio augmentations for robust training
- train_utils.py: Training utilities (metrics, early stopping, logging)
- fine_tune.py: Multi-phase fine-tuning orchestrator
"""

from training.dataset import CrossSpeciesDataset
from training.augmentations import MelAugmenter
from training.train_utils import (
    EarlyStopping,
    MetricsTracker,
    TrainingLogger,
    train_epoch,
    validate,
)
from training.fine_tune import FineTuner

__all__ = [
    "CrossSpeciesDataset",
    "MelAugmenter",
    "EarlyStopping",
    "MetricsTracker",
    "TrainingLogger",
    "train_epoch",
    "validate",
    "FineTuner",
]