# evaluation/benchmark.py
"""
Benchmark fine-tuned model against baselines.

Baselines:
1. Random classifier (lower bound)
2. Majority class classifier
3. Simple CNN trained from scratch (comparison point)
4. Whisper features + logistic regression (shallow baseline)

This demonstrates WHY fine-tuning is better than:
- Training from scratch (needs more data)
- Simple approaches (less accurate)
- Not using pre-trained features (worse on small data)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import Counter
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class RandomBaseline:
    """Predict classes randomly (absolute lower bound)."""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def predict(self, n_samples: int) -> np.ndarray:
        return np.random.randint(0, self.num_classes, n_samples)


class MajorityBaseline:
    """Always predict the most common class."""
    
    def __init__(self):
        self.majority_class = None
    
    def fit(self, labels: List[int]):
        counter = Counter(labels)
        self.majority_class = counter.most_common(1)[0][0]
    
    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, self.majority_class)


class SimpleCNNBaseline(nn.Module):
    """
    Simple CNN trained from scratch.
    
    This represents what you could build WITHOUT fine-tuning Whisper.
    Much smaller, but also much less accurate on limited data.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # x: [batch, n_mels, time] -> [batch, 1, n_mels, time]
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return {"logits": self.classifier(x)}


class BenchmarkRunner:
    """
    Compare model performance against baselines.
    
    Shows:
    - How much better fine-tuning is vs training from scratch
    - How pre-trained features improve data efficiency
    - Baseline comparison for context
    """
    
    def __init__(self, test_labels: List[int], num_classes: int):
        self.test_labels = test_labels
        self.num_classes = num_classes
        self.n_samples = len(test_labels)
        
        # Results storage
        self.results = {}
    
    def evaluate_predictions(self, preds: np.ndarray, name: str) -> Dict:
        """Compute metrics for a set of predictions."""
        labels = np.array(self.test_labels)
        accuracy = (preds == labels).mean()
        
        # Per-class accuracy
        per_class = {}
        for cls in range(self.num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                per_class[int(cls)] = (preds[mask] == cls).mean()
        
        return {
            "name": name,
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
        }
    
    def run_random_baseline(self) -> Dict:
        """Evaluate random classifier."""
        baseline = RandomBaseline(self.num_classes)
        accuracies = []
        
        # Run multiple times for stable estimate
        for _ in range(100):
            preds = baseline.predict(self.n_samples)
            acc = (preds == np.array(self.test_labels)).mean()
            accuracies.append(acc)
        
        result = {
            "name": "Random Baseline",
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "description": "Random guessing (lower bound)",
        }
        
        self.results["random"] = result
        return result
    
    def run_majority_baseline(self) -> Dict:
        """Evaluate majority class baseline."""
        baseline = MajorityBaseline()
        baseline.fit(self.test_labels)
        preds = baseline.predict(self.n_samples)
        accuracy = (preds == np.array(self.test_labels)).mean()
        
        result = {
            "name": "Majority Class Baseline",
            "accuracy": accuracy,
            "description": "Always predicts the most common class",
        }
        
        self.results["majority"] = result
        return result
    
    def run_model(self, model, test_loader, name: str, device: str = "cpu") -> Dict:
        """Evaluate a PyTorch model."""
        model = model.to(device)
        model.eval()
        
        all_preds = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                mel = batch["mel"].to(device)
                output = model(mel)
                preds = output["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
        
        total_time = time.time() - start_time
        
        result = self.evaluate_predictions(np.array(all_preds), name)
        result["inference_time_s"] = total_time
        result["samples_per_second"] = len(all_preds) / total_time
        
        self.results[name] = result
        return result
    
    def print_comparison(self):
        """Print comparison table."""
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)
        print(f"{'Model':<30s} {'Accuracy':>10s} {'Speed':>15s}")
        print("-" * 60)
        
        for name, result in self.results.items():
            if "accuracy_mean" in result:
                acc = f"{result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}"
            elif "accuracy" in result:
                acc = f"{result['accuracy']:.3f}"
            else:
                acc = "N/A"
            
            speed = ""
            if "samples_per_second" in result:
                speed = f"{result['samples_per_second']:.1f} samp/s"
            
            print(f"{result.get('name', name):<30s} {acc:>10s} {speed:>15s}")
        
        print("=" * 60)


# ─── Test ───
if __name__ == "__main__":
    # Simulate test labels
    num_classes = 20
    test_labels = np.random.randint(0, num_classes, 500).tolist()
    
    runner = BenchmarkRunner(test_labels, num_classes)
    
    runner.run_random_baseline()
    runner.run_majority_baseline()
    
    runner.print_comparison()