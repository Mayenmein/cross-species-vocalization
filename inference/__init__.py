# inference/__init__.py
"""
Inference module for cross-species vocalization model.

Contains:
- classifier.py: Single-file classification with detailed output
- live_mic.py: Real-time microphone capture and classification
- batch_process.py: Process entire folders of audio files
"""

from inference.classifier import AudioClassifier
from inference.live_mic import LiveMicrophoneClassifier
from inference.batch_process import BatchProcessor

__all__ = [
    "AudioClassifier",
    "LiveMicrophoneClassifier",
    "BatchProcessor",
]