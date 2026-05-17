"""
Confusion matrix visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import SOUND_CLASSES, DOMAIN_MAP


class ConfusionMatrixVisualizer:
    """
    Visualize model errors and confusion patterns.
    """
    
    def __init__(self, save_dir: str = "evaluation/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: List[List[int]],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (14, 12),
    ) -> str:
        """Plot confusion matrix heatmap."""
        cm = np.array(confusion_matrix)
        
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
            fmt = ".2f"
            cbar_label = "Proportion"
        else:
            fmt = "d"
            cbar_label = "Count"
        
        # Use class names (truncated for display)
        class_names = [
            SOUND_CLASSES.get(i, f"C{i}").replace("_", "\n")[:15]
            for i in range(cm.shape[0])
        ]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="RdYlBu_r",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": cbar_label},
            annot_kws={"size": 6},
        )
        
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        save_path = self.save_dir / f"confusion_matrix_{'norm' if normalize else 'count'}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved to {save_path}")
        return str(save_path)
    
    def plot_per_class_metrics(
        self,
        per_class_metrics: Dict,
        figsize: Tuple[int, int] = (14, 8),
    ) -> str:
        """Plot per-class precision, recall, F1."""
        classes = list(per_class_metrics.keys())
        precisions = [per_class_metrics[c]["precision"] for c in classes]
        recalls = [per_class_metrics[c]["recall"] for c in classes]
        f1s = [per_class_metrics[c]["f1"] for c in classes]
        supports = [per_class_metrics[c]["support"] for c in classes]
        
        # Sort by F1
        sorted_idx = np.argsort(f1s)[::-1]
        classes = [classes[i] for i in sorted_idx]
        precisions = [precisions[i] for i in sorted_idx]
        recalls = [recalls[i] for i in sorted_idx]
        f1s = [f1s[i] for i in sorted_idx]
        supports = [supports[i] for i in sorted_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]})
        
        # Bar chart
        x = np.arange(len(classes))
        width = 0.25
        
        ax1.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        ax1.bar(x, recalls, width, label="Recall", alpha=0.8)
        ax1.bar(x + width, f1s, width, label="F1 Score", alpha=0.8)
        
        ax1.set_ylabel("Score", fontsize=12)
        ax1.set_title("Per-Class Performance", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.replace("_", " ")[:20] for c in classes], rotation=45, ha="right", fontsize=8)
        ax1.legend(loc="lower right")
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)
        
        # Support bar chart
        colors = plt.cm.viridis(np.array(supports) / max(supports))
        ax2.barh(classes, supports, color=colors)
        ax2.set_xlabel("Number of Samples", fontsize=12)
        ax2.set_title("Class Support", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        save_path = self.save_dir / "per_class_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved to {save_path}")
        return str(save_path)
    
    def plot_domain_performance(self, per_domain_metrics: Dict) -> str:
        """Plot per-domain accuracy."""
        domains = list(per_domain_metrics.keys())
        accuracies = [per_domain_metrics[d]["accuracy"] for d in domains]
        counts = [per_domain_metrics[d]["num_samples"] for d in domains]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(
            [d.replace("_", " ").title() for d in domains],
            accuracies,
            color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
            alpha=0.8,
            edgecolor="black",
        )
        
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.2%}",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Performance by Domain", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        
        # Subtitle with sample counts
        subtitle = " | ".join(f"{d}: {c}" for d, c in zip(domains, counts))
        ax.set_xlabel(subtitle, fontsize=9, color="gray")
        
        plt.tight_layout()
        
        save_path = self.save_dir / "domain_performance.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved to {save_path}")
        return str(save_path)


if __name__ == "__main__":
    # Test with dummy data
    num_classes = 30
    cm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        cm[i, i] = 50
        cm[i, (i + 1) % num_classes] = 10
    
    viz = ConfusionMatrixVisualizer()
    viz.plot_confusion_matrix(cm.tolist())
    print("Test complete")