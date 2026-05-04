# evaluation/confusion_matrix.py
"""
Confusion matrix visualization and error analysis.

Generates:
- Confusion matrix heatmap
- Per-class error breakdown
- Domain confusion analysis
- Attention overlay on spectrograms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import SOUND_CLASSES, DOMAIN_MAP


class ConfusionMatrixVisualizer:
    """
    Visualize model errors and confusion patterns.
    
    Helps understand:
    - Which classes are confused with each other
    - Whether errors stay within domains or cross domains
    - Where the model is overconfident vs underconfident
    """
    
    def __init__(self, save_dir: str = "evaluation/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: List[List[int]],
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (14, 12),
    ) -> str:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            confusion_matrix: 2D list of counts
            class_names: Names for each class (from SOUND_CLASSES)
            normalize: If True, show percentages instead of counts
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved figure
        """
        cm = np.array(confusion_matrix)
        
        if class_names is None:
            class_names = [
                SOUND_CLASSES.get(i, f"C{i}").replace("_", " ")
                for i in range(cm.shape[0])
            ]
        
        if normalize:
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        else:
            cm_norm = cm
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Custom colormap: white for 0, dark for high values
        colors = [(1, 1, 1), (0.2, 0.4, 0.8), (0.8, 0.1, 0.1)]
        cmap = LinearSegmentedColormap.from_list("custom_cm", colors, N=256)
        
        # Heatmap
        im = ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Add text annotations for high values
        if normalize:
            threshold = 0.3
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if cm_norm[i, j] > threshold:
                        text_color = "white" if cm_norm[i, j] > 0.7 else "black"
                        ax.text(
                            j, i, f"{cm_norm[i, j]:.2f}",
                            ha="center", va="center",
                            fontsize=6, color=text_color,
                        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Proportion" if normalize else "Count", fontsize=10)
        
        plt.tight_layout()
        
        # Save
        save_path = self.save_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Confusion matrix saved to {save_path}")
        return str(save_path)
    
    def plot_per_class_errors(
        self,
        per_class_metrics: Dict,
        figsize: Tuple[int, int] = (14, 8),
    ) -> str:
        """
        Plot per-class precision, recall, and F1 scores.
        
        Args:
            per_class_metrics: Dict from ModelEvaluator results
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        class_names = list(per_class_metrics.keys())
        precisions = [m["precision"] for m in per_class_metrics.values()]
        recalls = [m["recall"] for m in per_class_metrics.values()]
        f1s = [m["f1"] for m in per_class_metrics.values()]
        supports = [m["support"] for m in per_class_metrics.values()]
        
        # Sort by F1 score
        sorted_idx = np.argsort(f1s)[::-1]
        class_names = [class_names[i].replace("_", " ") for i in sorted_idx]
        precisions = [precisions[i] for i in sorted_idx]
        recalls = [recalls[i] for i in sorted_idx]
        f1s = [f1s[i] for i in sorted_idx]
        supports = [supports[i] for i in sorted_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})
        
        # Bar chart of metrics
        x = np.arange(len(class_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, precisions, width, label="Precision", color="#2ecc71", alpha=0.8)
        bars2 = ax1.bar(x, recalls, width, label="Recall", color="#3498db", alpha=0.8)
        bars3 = ax1.bar(x + width, f1s, width, label="F1 Score", color="#e74c3c", alpha=0.8)
        
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax1.legend(loc="lower right", fontsize=10)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)
        
        # Support (sample count) per class
        colors = plt.cm.viridis(np.array(supports) / max(supports))
        ax2.barh(class_names, supports, color=colors)
        ax2.set_xlabel("Number of Samples", fontsize=12)
        ax2.set_title("Class Support", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        save_path = self.save_dir / "per_class_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Per-class metrics saved to {save_path}")
        return str(save_path)
    
    def plot_domain_performance(
        self,
        per_domain_metrics: Dict,
        figsize: Tuple[int, int] = (8, 6),
    ) -> str:
        """
        Plot per-domain accuracy comparison.
        
        Args:
            per_domain_metrics: Dict from ModelEvaluator results
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        domains = list(per_domain_metrics.keys())
        accuracies = [m["accuracy"] for m in per_domain_metrics.values()]
        sample_counts = [m["num_samples"] for m in per_domain_metrics.values()]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        bars = ax.bar(
            [d.replace("_", " ").title() for d in domains],
            accuracies,
            color=colors[:len(domains)],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.2%}",
                ha="center",
                fontsize=14,
                fontweight="bold",
            )
        
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Performance by Domain", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        
        # Add sample count as subtitle
        subtitle = " | ".join(
            f"{d.replace('_', ' ').title()}: {c} samples"
            for d, c in zip(domains, sample_counts)
        )
        ax.set_xlabel(subtitle, fontsize=9, color="gray")
        
        plt.tight_layout()
        
        save_path = self.save_dir / "domain_performance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Domain performance saved to {save_path}")
        return str(save_path)
    
    def plot_confidence_distribution(
        self,
        correct_confidences: List[float],
        incorrect_confidences: List[float],
        figsize: Tuple[int, int] = (10, 6),
    ) -> str:
        """
        Plot confidence distribution for correct vs incorrect predictions.
        
        Good models are:
        - Confident when correct (peak near 1.0)
        - Uncertain when incorrect (spread out, not peak near 1.0)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        bins = np.linspace(0, 1, 21)
        
        ax.hist(
            correct_confidences, bins=bins, alpha=0.6,
            label=f"Correct (n={len(correct_confidences)})",
            color="#2ecc71", edgecolor="black",
        )
        ax.hist(
            incorrect_confidences, bins=bins, alpha=0.6,
            label=f"Incorrect (n={len(incorrect_confidences)})",
            color="#e74c3c", edgecolor="black",
        )
        
        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Model Confidence: Correct vs Incorrect Predictions", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / "confidence_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Confidence distribution saved to {save_path}")
        return str(save_path)


# ─── Test ───
if __name__ == "__main__":
    # Create dummy data for testing
    num_classes = 20
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Simulate mostly diagonal (correct) with some confusion
    for i in range(num_classes):
        cm[i, i] = np.random.randint(30, 100)
        # Confuse with neighboring class
        cm[i, (i + 1) % num_classes] = np.random.randint(0, 15)
        cm[i, (i - 1) % num_classes] = np.random.randint(0, 10)
    
    viz = ConfusionMatrixVisualizer()
    viz.plot_confusion_matrix(cm.tolist())
    print("Test visualization complete.")