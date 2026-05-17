"""
Benchmark fine-tuned model against baselines.
"""

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import SOUND_CLASSES


class RandomBaseline:
    """Random classifier (lower bound)."""
    def predict(self, labels: List[int]) -> np.ndarray:
        return np.random.randint(0, max(labels) + 1, len(labels))


class MajorityBaseline:
    """Always predict most common class."""
    def fit(self, labels: List[int]):
        self.majority = Counter(labels).most_common(1)[0][0]
    
    def predict(self, labels: List[int]) -> np.ndarray:
        return np.full(len(labels), self.majority)


class BenchmarkRunner:
    """Compare model against baselines."""
    
    def __init__(self, test_labels: List[int]):
        self.test_labels = np.array(test_labels)
        self.num_classes = max(test_labels) + 1
    
    def run_baselines(self) -> Dict:
        """Run baseline comparisons."""
        results = {}
        
        # Random baseline
        random_clf = RandomBaseline()
        random_preds = random_clf.predict(self.test_labels)
        results["random"] = {
            "name": "Random Baseline",
            "accuracy": accuracy_score(self.test_labels, random_preds),
            "description": "Random guessing",
        }
        
        # Majority baseline
        majority_clf = MajorityBaseline()
        majority_clf.fit(self.test_labels.tolist())
        majority_preds = majority_clf.predict(self.test_labels)
        results["majority"] = {
            "name": "Majority Class Baseline",
            "accuracy": accuracy_score(self.test_labels, majority_preds),
            "description": "Always predict most common class",
        }
        
        return results
    
    def evaluate_model(self, predictions: List[int], name: str = "Fine-tuned Model") -> Dict:
        """Evaluate a model's predictions."""
        preds = np.array(predictions)
        
        # Overall accuracy
        accuracy = accuracy_score(self.test_labels, preds)
        
        # Per-class accuracy
        per_class = {}
        for cls in range(self.num_classes):
            mask = self.test_labels == cls
            if mask.sum() > 0:
                per_class[SOUND_CLASSES.get(cls, f"class_{cls}")] = (preds[mask] == cls).mean()
        
        return {
            "name": name,
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
        }
    
    def print_comparison(self, model_results: Dict, baseline_results: Dict):
        """Print comparison table."""
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)
        print(f"{'Model':<30s} {'Accuracy':>10s}")
        print("-" * 60)
        
        for name, result in baseline_results.items():
            print(f"{result['name']:<30s} {result['accuracy']:>9.3f}")
        
        print(f"{model_results['name']:<30s} {model_results['accuracy']:>9.3f}")
        
        print("=" * 60)
        print("\nImprovement over random: "
              f"{(model_results['accuracy'] - baseline_results['random']['accuracy'])*100:.1f}%")
        print("Improvement over majority: "
              f"{(model_results['accuracy'] - baseline_results['majority']['accuracy'])*100:.1f}%")


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    test_labels = np.random.randint(0, 20, 500).tolist()
    
    runner = BenchmarkRunner(test_labels)
    baseline_results = runner.run_baselines()
    
    # Dummy model predictions (would come from actual model)
    dummy_preds = np.random.randint(0, 20, 500).tolist()
    model_results = runner.evaluate_model(dummy_preds)
    
    runner.print_comparison(model_results, baseline_results)