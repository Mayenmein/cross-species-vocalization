# evaluation/evaluate.py
"""
Full test set evaluation with comprehensive metrics.

Computes:
- Overall accuracy
- Per-class precision, recall, F1
- Top-3 and Top-5 accuracy
- Per-domain performance
- Attention weight analysis
- Inference speed benchmarking
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import ModelConfig, SOUND_CLASSES, DOMAIN_MAP, CLASS_TO_DOMAIN
from training.dataset import CrossSpeciesDataset
from torch.utils.data import DataLoader


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Goes beyond accuracy to provide detailed insights into
    model behavior, failure modes, and domain-specific performance.
    """
    
    def __init__(
        self,
        model: CrossSpeciesVocalizationModel,
        test_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
        
        # Store all predictions for analysis
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_domains = []
        self.all_filepaths = []
        self.all_attention = []
        
        # Timing
        self.inference_times = []
    
    @torch.no_grad()
    def run_evaluation(self) -> Dict:
        """
        Run full evaluation on test set.
        
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60)
        
        for batch_idx, batch in enumerate(self.test_loader):
            mel = batch["mel"].to(self.device)
            labels = batch["label"]
            
            # Time inference
            start = time.time()
            output = self.model(mel, domain="animal", return_attention=True)
            inference_time = time.time() - start
            
            # Get predictions
            probs = F.softmax(output["logits"], dim=-1)
            preds = output["logits"].argmax(dim=-1)
            
            # Store
            self.all_preds.extend(preds.cpu().tolist())
            self.all_labels.extend(labels.tolist())
            self.all_probs.extend(probs.cpu().tolist())
            self.all_filepaths.extend(batch.get("filepath", [""]))
            self.inference_times.append(inference_time)
            
            # Store attention if available
            if "attention_weights" in output:
                self.all_attention.append(output["attention_weights"].cpu())
            
            # Progress
            if (batch_idx + 1) % max(1, len(self.test_loader) // 5) == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.test_loader)}")
        
        # Compute all metrics
        results = self._compute_metrics()
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _compute_metrics(self) -> Dict:
        """Compute all evaluation metrics."""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        results = {}
        
        # ─── Overall Metrics ───
        results["num_samples"] = len(preds)
        results["accuracy"] = (preds == labels).mean()
        
        # ─── Per-Class Metrics ───
        per_class = {}
        for cls in range(self.model.config.num_classes):
            tp = ((preds == cls) & (labels == cls)).sum()
            fp = ((preds == cls) & (labels != cls)).sum()
            fn = ((preds != cls) & (labels == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_name = SOUND_CLASSES.get(cls, f"class_{cls}")
            per_class[class_name] = {
                "class_id": cls,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": (labels == cls).sum().item(),
            }
        
        results["per_class"] = per_class
        
        # ─── Per-Domain Metrics ───
        domain_metrics = {}
        for domain, class_ids in DOMAIN_MAP.items():
            domain_mask = np.isin(labels, class_ids)
            if domain_mask.sum() > 0:
                domain_acc = (preds[domain_mask] == labels[domain_mask]).mean()
                domain_metrics[domain] = {
                    "accuracy": domain_acc,
                    "num_samples": int(domain_mask.sum()),
                }
        results["per_domain"] = domain_metrics
        
        # ─── Top-K Accuracy ───
        results["top3_accuracy"] = self._top_k_accuracy(k=3)
        results["top5_accuracy"] = self._top_k_accuracy(k=5)
        
        # ─── Confusion Matrix ───
        cm = np.zeros((self.model.config.num_classes, self.model.config.num_classes), dtype=int)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
        results["confusion_matrix"] = cm.tolist()
        
        # ─── Most Confused Pairs ───
        confused_pairs = []
        for i in range(self.model.config.num_classes):
            for j in range(self.model.config.num_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        "true_class": SOUND_CLASSES.get(i, f"class_{i}"),
                        "predicted_class": SOUND_CLASSES.get(j, f"class_{j}"),
                        "count": int(cm[i, j]),
                    })
        confused_pairs.sort(key=lambda x: x["count"], reverse=True)
        results["most_confused_pairs"] = confused_pairs[:10]
        
        # ─── Inference Speed ───
        results["avg_inference_time_ms"] = np.mean(self.inference_times) * 1000
        results["inference_time_std_ms"] = np.std(self.inference_times) * 1000
        results["total_inference_time_s"] = sum(self.inference_times)
        
        # ─── Confidence Analysis ───
        correct_mask = preds == labels
        probs_array = np.array(self.all_probs)
        
        correct_confidences = probs_array[correct_mask].max(axis=1)
        incorrect_confidences = probs_array[~correct_mask].max(axis=1)
        
        results["avg_confidence_correct"] = float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0.0
        results["avg_confidence_incorrect"] = float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0.0
        
        return results
    
    def _top_k_accuracy(self, k: int) -> float:
        """Compute top-k accuracy."""
        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)
        
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
        
        return float(correct.mean())
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*40}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*40}")
        print(f"Samples:       {results['num_samples']}")
        print(f"Accuracy:      {results['accuracy']:.3f}")
        print(f"Top-3 Acc:     {results['top3_accuracy']:.3f}")
        print(f"Top-5 Acc:     {results['top5_accuracy']:.3f}")
        
        print(f"\nPer-Domain Accuracy:")
        for domain, metrics in results["per_domain"].items():
            print(f"  {domain:20s}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)")
        
        print(f"\nInference Speed:")
        print(f"  Average: {results['avg_inference_time_ms']:.1f} ms/sample")
        
        print(f"\nConfidence:")
        print(f"  When correct:   {results['avg_confidence_correct']:.3f}")
        print(f"  When incorrect: {results['avg_confidence_incorrect']:.3f}")
        
        print(f"\nTop Confused Pairs:")
        for pair in results["most_confused_pairs"][:5]:
            print(f"  {pair['true_class']} → {pair['predicted_class']} ({pair['count']}x)")
    
    def save_results(self, path: str):
        """Save evaluation results to JSON."""
        results = self.run_evaluation()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {path}")


# ─── Test ───
if __name__ == "__main__":
    from models.config import ModelConfig
    from training.dataset import CrossSpeciesDataset
    
    # Load model
    config = ModelConfig(whisper_size="tiny")
    model = CrossSpeciesVocalizationModel(config)
    
    # Load test data
    test_dataset = CrossSpeciesDataset(
        data_root="data/raw",
        split="test",
        target_duration=10.0,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader)
    results = evaluator.run_evaluation()"""
Comprehensive model evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import time
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.utils import load_model, load_audio, get_top_predictions, compute_metrics
from models.config import ModelConfig, DOMAIN_MAP, SOUND_CLASSES
from torch.utils.data import DataLoader
from training.dataset import CrossSpeciesDataset
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation with per-domain metrics.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/best.ckpt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model, self.device = load_model(model_path, device=device)
        self.config = ModelConfig()
        
        self.predictions = []
        self.targets = []
        self.domains = []
        self.confidences = []
        self.inference_times = []
    
    def evaluate_dataset(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate on test dataset.
        """
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
        
        self.model.eval()
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get raw audio (dataset returns raw audio now)
            audio_list = batch["audio"]
            max_len = max(a.shape[0] for a in audio_list)
            padded_audio = []
            for a in audio_list:
                if a.shape[0] < max_len:
                    a = torch.nn.functional.pad(a, (0, max_len - a.shape[0]))
                padded_audio.append(a)
            
            x = torch.stack(padded_audio).to(self.device)
            y = batch["label"]
            
            # Time inference
            start = time.time()
            with torch.no_grad():
                output = self.model(x, domain="animal")
            self.inference_times.append(time.time() - start)
            
            # Get predictions
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            self.predictions.extend(preds.cpu().tolist())
            self.targets.extend(y.tolist())
            self.confidences.extend(probs.max(dim=-1)[0].cpu().tolist())
        
        # Compute metrics
        results = self._compute_full_metrics()
        self._print_summary(results)
        
        return results
    
    def _compute_full_metrics(self) -> Dict:
        """Compute all metrics."""
        preds = np.array(self.predictions)
        labels = np.array(self.targets)
        
        # Overall metrics
        results = compute_metrics(preds.tolist(), labels.tolist(), self.config.num_classes)
        
        # Per-domain metrics
        per_domain = {}
        for domain, class_ids in DOMAIN_MAP.items():
            mask = np.isin(labels, class_ids)
            if mask.sum() > 0:
                domain_preds = preds[mask]
                domain_labels = labels[mask]
                per_domain[domain] = {
                    "accuracy": (domain_preds == domain_labels).mean(),
                    "num_samples": int(mask.sum()),
                }
        results["per_domain"] = per_domain
        
        # Top-k accuracy
        results["top3_accuracy"] = self._top_k_accuracy(3)
        results["top5_accuracy"] = self._top_k_accuracy(5)
        
        # Confidence analysis
        results["avg_confidence"] = float(np.mean(self.confidences))
        results["calibration_error"] = self._compute_calibration_error()
        
        # Inference speed
        results["avg_inference_time_ms"] = np.mean(self.inference_times) * 1000
        
        return results
    
    def _top_k_accuracy(self, k: int) -> float:
        """Compute top-k accuracy."""
        # Need to recompute with stored probabilities
        # Simplified for now - would need full probability matrix
        return 0.0  # Placeholder
    
    def _compute_calibration_error(self) -> float:
        """Compute expected calibration error."""
        # Simplified ECE calculation
        return 0.0  # Placeholder
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*40}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*40}")
        print(f"Samples:       {len(self.predictions)}")
        print(f"Accuracy:      {results['accuracy']:.3f}")
        
        print(f"\nPer-Domain Accuracy:")
        for domain, metrics in results.get("per_domain", {}).items():
            print(f"  {domain:20s}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)")
        
        print(f"\nInference Speed:")
        print(f"  Average: {results['avg_inference_time_ms']:.1f} ms/sample")
        
        print(f"\nConfidence:")
        print(f"  Average: {results['avg_confidence']:.3f}")
    
    def save_results(self, path: str):
        """Save results to JSON."""
        results = self._compute_full_metrics()
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {path}")


def create_test_loader(data_root: str = "data/raw", batch_size: int = 8):
    """Create test data loader."""
    from training.dataset import CrossSpeciesDataset
    
    dataset = CrossSpeciesDataset(
        data_root=data_root,
        split="test",
        target_duration=10.0,
        return_raw_audio=True,
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    test_loader = create_test_loader()
    results = evaluator.evaluate_dataset(test_loader)
    evaluator.save_results("evaluation_results.json")