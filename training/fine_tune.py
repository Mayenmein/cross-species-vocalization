# training/fine_tune.py
"""
Fine-tuning pipeline for Cross-Species Vocalization Model.

Three-phase training:
1. Train new heads with frozen Whisper
2. Unfreeze top layers for domain adaptation
3. Train domain-specific adapters
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable, List
from contextlib import nullcontext

from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import ModelConfig, TrainingConfig, SOUND_CLASSES, DOMAINS
from training.dataset import CrossSpeciesDataset, collate_fn
from training.augmentations import MelAugmenter
from training.train_utils import (
    EarlyStopping,
    MetricsTracker,
    TrainingLogger,
    train_epoch,
    validate,
)
import time

class _CollateWithAugmenter:
    """Picklable collate function wrapper for Windows multiprocessing."""
    def __init__(self, mel_augmenter=None):
        self.mel_augmenter = mel_augmenter
    
    def __call__(self, batch):
        return collate_fn(batch, self.mel_augmenter)
# ---------------------------
# Utility: build dataloaders
# ---------------------------
def build_loaders(model_config, training_config):
    """Build train/val/test dataloaders with optional augmentation."""
    
    # Create mel augmenter for training
    mel_augmenter = None
    if training_config.use_augmentation:
        mel_augmenter = MelAugmenter(
            sample_rate=model_config.sample_rate,
            n_mels=model_config.n_mels,
            apply_prob=training_config.augmentation_prob,
        )
    
    def make(split, aug):
        """Create dataset for a given split."""
        return CrossSpeciesDataset(
            data_root=training_config.data_root,
            split=split,
            target_duration=model_config.target_duration,
            sample_rate=model_config.sample_rate,
            n_mels=model_config.n_mels,
            augment=aug,
            augment_prob=training_config.augmentation_prob,
            mel_augmenter=mel_augmenter if aug else None,
            return_raw_audio=True,  # Let augmenter handle mel conversion
        )

    train_ds = make("train", training_config.use_augmentation)
    val_ds = make("val", False)
    test_ds = make("test", False)

    # Use custom collate for proper batching
    train_collate = _CollateWithAugmenter(mel_augmenter if training_config.use_augmentation else None)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.phase1_batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=train_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.phase1_batch_size * 2,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=training_config.phase1_batch_size * 2,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, mel_augmenter


# ---------------------------
# FineTuner
# ---------------------------
class FineTuner:
    """
    Orchestrates the three-phase fine-tuning process.
    """
    
    def __init__(self, model, model_config: ModelConfig, training_config: TrainingConfig):
        self.model = model
        self.cfg = training_config
        self.mcfg = model_config

        self.device = torch.device(training_config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build loaders and augmenter
        self.train_loader, self.val_loader, self.test_loader, self.mel_augmenter = \
            build_loaders(model_config, training_config)

        self.logger = TrainingLogger(
            log_dir=training_config.log_dir,
            use_wandb=training_config.use_wandb,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.current_phase = 0

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Train samples: {len(self.train_loader.dataset)}")
        print(f"[INFO] Val samples: {len(self.val_loader.dataset)}")
        print(f"[INFO] Test samples: {len(self.test_loader.dataset)}")

    # ---------------------------
    # Core training phase logic
    # ---------------------------
    def _run_phase(
        self,
        phase_id: int,
        epochs: int,
        optimizer_fn: Callable,
        scheduler_fn: Optional[Callable],
        early_stop_path: str,
        domain: Optional[str] = None,
    ):

        print(f"\n{'='*60}")
        print(f"PHASE {phase_id}")
        print(f"{'='*60}")

        print(f"Starting Phase {phase_id} at {time.strftime('%H:%M:%S')}")

        optimizer = optimizer_fn()
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            save_path=early_stop_path,
        )

        # Create metrics tracker
        num_classes = self.mcfg.num_classes
        train_metrics = MetricsTracker(num_classes)
        val_metrics = MetricsTracker(num_classes)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            # Training
            train_loss = train_epoch(
                self.model,
                self.train_loader,
                optimizer,
                self.criterion,
                device=self.device,
                use_amp=self.cfg.use_amp,
                gradient_clip_norm=self.cfg.gradient_clip_norm,
                domain=domain or "animal",
            )

            # Validation
            val_loss, val_acc, val_metrics_dict = validate(
                self.model,
                self.val_loader,
                self.criterion,
                device=self.device,
                domain=domain or "animal",
            )

            lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            print(f"  Epoch completed in {epoch_time:.1f}s")

            # Log
            self.logger.log_epoch(
                phase=phase_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=lr,
            )

            # Scheduler step
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if early_stopping(val_loss, self.model, epoch):
                print("[EARLY STOP] Triggered")
                break

        # Load best checkpoint
        self._load(early_stop_path)

    # ---------------------------
    # Phase 1: Train new heads only
    # ---------------------------
    def phase1(self):
        """Train temporal aggregator, attention, classifier, and domain classifier."""
        print("\n[PHASE 1] Training new heads with frozen Whisper encoder")
        
        # Freeze Whisper
        self.model._freeze_whisper()

        def optimizer_fn():
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg.phase1_lr,
                weight_decay=self.cfg.weight_decay,
            )

        def scheduler_fn(opt):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=self.cfg.phase1_epochs,
                eta_min=self.cfg.phase1_lr / 100,
            )

        # Enable only new modules
        self._enable_modules([
            "temporal_aggregator",
            "attention",
            "query_token",
            "classifier",
            "domain_classifier",
            "projection",
            "ssl_head",
        ])

        self._run_phase(
            phase_id=1,
            epochs=self.cfg.phase1_epochs,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            early_stop_path=f"{self.cfg.checkpoint_dir}/phase1.pt",
        )

    # ---------------------------
    # Phase 2: Unfreeze top encoder layers
    # ---------------------------
    def phase2(self):
        """Unfreeze top Whisper layers and continue training."""
        print(f"\n[PHASE 2] Unfreezing top {self.cfg.phase2_layers_to_unfreeze} Whisper layers")
        
        # Unfreeze top layers
        self.model._unfreeze_top_layers(self.cfg.phase2_layers_to_unfreeze)

        # Keep all new modules trainable
        self._enable_modules([
            "temporal_aggregator",
            "attention",
            "query_token",
            "classifier",
            "domain_classifier",
            "projection",
            "ssl_head",
        ])

        def optimizer_fn():
            # Differential learning rates
            encoder_params = [p for n, p in self.model.named_parameters() 
                            if "whisper" in n or "encoder" in n and p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() 
                          if "whisper" not in n and "encoder" not in n and p.requires_grad]
            
            return torch.optim.AdamW([
                {"params": encoder_params, "lr": self.cfg.phase2_lr},
                {"params": other_params, "lr": self.cfg.phase2_lr * 10},
            ], weight_decay=self.cfg.weight_decay)

        def scheduler_fn(opt):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=3, factor=0.5
            )

        self._run_phase(
            phase_id=2,
            epochs=self.cfg.phase2_epochs,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            early_stop_path=f"{self.cfg.checkpoint_dir}/phase2.pt",
        )

    # ---------------------------
    # Phase 3: Train domain adapters
    # ---------------------------
    def phase3(self):
        """Train domain-specific adapters individually."""
        if not self.model.use_domain_adapters:
            print("[SKIP] No domain adapters enabled")
            return

        print(f"\n[PHASE 3] Training domain-specific adapters")

        # Freeze everything except adapters
        for p in self.model.parameters():
            p.requires_grad = False

        for domain in DOMAINS:
            print(f"\n[ADAPTER] Training adapter for {domain}")
            
            # Enable only this domain's adapter
            for p in self.model.domain_adapters[domain].parameters():
                p.requires_grad = True
            
            # Enable classifier for fine-tuning
            for p in self.model.classifier.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                [
                    {"params": self.model.domain_adapters[domain].parameters()},
                    {"params": self.model.classifier.parameters(), "lr": self.cfg.phase3_lr * 0.1},
                ],
                lr=self.cfg.phase3_lr,
                weight_decay=self.cfg.weight_decay,
            )

            for epoch in range(1, self.cfg.phase3_epochs + 1):
                train_loss = train_epoch(
                    self.model,
                    self.train_loader,
                    optimizer,
                    self.criterion,
                    device=self.device,
                    use_amp=self.cfg.use_amp,
                    gradient_clip_norm=self.cfg.gradient_clip_norm,
                    domain=domain,
                )
                print(f"  Epoch {epoch}/{self.cfg.phase3_epochs} - Loss: {train_loss:.4f}")

            # Freeze back
            for p in self.model.domain_adapters[domain].parameters():
                p.requires_grad = False

        # Save final checkpoint
        checkpoint_path = f"{self.cfg.checkpoint_dir}/phase3_final.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.mcfg,
        }, checkpoint_path)
        print(f"\n[SAVED] Final model to {checkpoint_path}")

    # ---------------------------
    # Evaluation
    # ---------------------------
    def evaluate(self, domain="animal"):
        """Evaluate the model on test set."""
        print(f"\n{'='*60}")
        print(f"TEST EVALUATION (domain: {domain})")
        print(f"{'='*60}")
        
        val_loss, val_acc, metrics = validate(
            self.model,
            self.test_loader,
            self.criterion,
            device=self.device,
            domain=domain,
        )

        print(f"\nTest Loss: {val_loss:.4f}")
        print(f"Test Accuracy: {val_acc:.3f}")

        if "per_class_accuracy" in metrics:
            print("\nPer-class Accuracy:")
            for class_id, acc in metrics["per_class_accuracy"].items():
                class_name = SOUND_CLASSES.get(int(class_id), f"Class {class_id}")
                print(f"  {class_name:30s}: {acc:.3f}")

        return metrics

    # ---------------------------
    # Helpers
    # ---------------------------
    def _enable_modules(self, keywords: List[str]):
        """Enable gradients only for modules containing any of the keywords."""
        for n, p in self.model.named_parameters():
            p.requires_grad = any(k in n for k in keywords)

    def _load(self, path: str):
        """Load checkpoint if exists."""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[LOADED] Checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            print(f"[WARNING] No checkpoint found at {path}")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    from models.config import get_configs
    
    # Load configs
    model_cfg, train_cfg = get_configs("tiny")
    
    # Create model
    model = CrossSpeciesVocalizationModel(model_cfg)
    
    # Create fine-tuner
    ft = FineTuner(model, model_cfg, train_cfg)
    
    # Run training phases
    print("\n" + "="*60)
    print("STARTING CROSS-SPECIES VOCALIZATION TRAINING")
    print("="*60)
    
    # Phase 1: Train new heads
    ft.phase1()
    
    # Phase 2: Unfreeze top layers
    ft.phase2()
    
    # Phase 3: Domain adapters
    ft.phase3()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    for domain in DOMAINS:
        ft.evaluate(domain)
    
    print("\nTraining complete!")