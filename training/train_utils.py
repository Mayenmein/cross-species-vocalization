"""
Robust training utilities for audio classification.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from contextlib import nullcontext


# ==================================================
# EARLY STOPPING
# ==================================================
class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
        mode: str = "min",
        save_path: str = "checkpoints/best.pt",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score = None
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module, epoch: int):
        if self.best_score is None:
            self._save(model, epoch, score)
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._save(model, epoch, score)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _save(self, model, epoch, score):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "score": score,
            },
            self.save_path,
        )


# ==================================================
# METRICS
# ==================================================
class MetricsTracker:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []
        self.losses = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        self.preds.append(logits.argmax(dim=-1).detach().cpu())
        self.targets.append(targets.detach().cpu())
        self.losses.append(loss)

    def compute(self) -> Dict:
        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()

        acc = (preds == targets).mean()
        avg_loss = float(np.mean(self.losses))

        # confusion matrix (vectorized)
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        np.add.at(cm, (targets, preds), 1)

        # per-class accuracy
        per_class = {}
        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class[c] = (preds[mask] == c).mean()

        return {
            "accuracy": float(acc),
            "avg_loss": avg_loss,
            "confusion_matrix": cm,
            "per_class_accuracy": per_class,
        }


# ==================================================
# LOGGER
# ==================================================
class TrainingLogger:
    def __init__(self, log_dir="logs", use_wandb=False, name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb

        self.name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.log_dir / self.name
        self.dir.mkdir(exist_ok=True)

        self.file = self.dir / "log.jsonl"
        self.history = []

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling wandb logging.")
                self.use_wandb = False

    def log_epoch(self, phase, epoch, train_loss, val_loss, val_accuracy, learning_rate):
        """Log training metrics for an epoch."""
        entry = {
            "time": time.time(),
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        }

        self.history.append(entry)

        with open(self.file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"[Phase {phase}][Epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.3f} "
            f"lr={learning_rate:.2e}"
        )

        if self.use_wandb:
            self.wandb.log(entry)

    def log(self, **kwargs):
        """Generic logging method."""
        entry = {
            "time": time.time(),
            **self._safe(kwargs),
        }

        self.history.append(entry)

        with open(self.file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _safe(self, d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out


# ==================================================
# TRAIN LOOP
# ==================================================
def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp=False,
    gradient_clip_norm=1.0,
    domain="animal",
):
    model.train()
    losses = []

    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    total_batches = len(loader)
    print(f"  Training on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        # Progress indicator every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{total_batches}")
        
        audio_list = batch["audio"]
        max_len = max(a.shape[0] for a in audio_list)
        padded_audio = []
        for a in audio_list:
            if a.shape[0] < max_len:
                a = torch.nn.functional.pad(a, (0, max_len - a.shape[0]))
            padded_audio.append(a)
        x = torch.stack(padded_audio).to(device)
        
        y = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda') if scaler else nullcontext():
            out = model(x, domain=domain)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))

# ==================================================
# VALIDATION
# ==================================================
@torch.no_grad()
def validate(model, loader, criterion, device, domain="animal"):
    model.eval()

    losses = []
    preds_all, labels_all = [], []
    
    total_batches = len(loader)
    print(f"  Validating on {total_batches} batches...")

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{total_batches}")
        
        audio_list = batch["audio"]
        max_len = max(a.shape[0] for a in audio_list)
        padded_audio = []
        for a in audio_list:
            if a.shape[0] < max_len:
                a = torch.nn.functional.pad(a, (0, max_len - a.shape[0]))
            padded_audio.append(a)
        x = torch.stack(padded_audio).to(device)
        
        y = batch["label"].to(device)

        out = model(x, domain=domain)
        logits = out["logits"] if isinstance(out, dict) else out

        loss = criterion(logits, y)
        losses.append(loss.item())

        preds_all.append(logits.argmax(-1).cpu())
        labels_all.append(y.cpu())

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)

    acc = (preds == labels).float().mean().item()

    per_class_acc = {}
    unique_labels = labels.unique()
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            per_class_acc[int(label)] = (preds[mask] == label).float().mean().item()

    metrics = {
        "accuracy": acc,
        "per_class_accuracy": per_class_acc
    }

    return float(np.mean(losses)), acc, metrics