# training/fine_tune.py
"""
DEPRECATED: This file is being replaced by lightning_module.py and run_training.py.
Kept for reference but no longer used in active training.

Fine-tuning pipeline with QLoRA.
Phase 1: Pre-compute embeddings, train heads
Phase 2: Unfreeze LoRA, train adapters + heads
Phase 3: Domain-specific LoRA training
""" 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import   DOMAINS
from training.dataset import CrossSpeciesDataset, collate_fn
from training.augmentations import MelAugmenter
from training.train_utils import ( TrainingLogger, train_epoch, validate)


class _CollateWithAugmenter:
    def __init__(self, mel_augmenter=None):
        self.mel_augmenter = mel_augmenter
    def __call__(self, batch):
        return collate_fn(batch, self.mel_augmenter)


def build_loaders(model_config, training_config):
    """Build dataloaders."""
    mel_augmenter = None
    if training_config.use_augmentation:
        mel_augmenter = MelAugmenter(
            sample_rate=model_config.sample_rate,
            n_mels=model_config.n_mels,
            apply_prob=training_config.augmentation_prob,
        )
    
    def make(split, aug):
        return CrossSpeciesDataset(
            data_root=training_config.data_root,
            split=split,
            target_duration=model_config.target_duration,
            sample_rate=model_config.sample_rate,
            n_mels=model_config.n_mels,
            augment=aug,
            augment_prob=training_config.augmentation_prob,
            mel_augmenter=mel_augmenter if aug else None,
            return_raw_audio=True,
        )

    train_ds = make("train", training_config.use_augmentation)
    val_ds = make("val", False)
    test_ds = make("test", False)

    train_collate = _CollateWithAugmenter(mel_augmenter if training_config.use_augmentation else None)
    
    train_loader = DataLoader(train_ds, batch_size=training_config.phase1_batch_size,
                              shuffle=True, num_workers=training_config.num_workers,
                              pin_memory=True, collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=training_config.phase1_batch_size * 2,
                            shuffle=False, num_workers=training_config.num_workers,
                            pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=training_config.phase1_batch_size * 2,
                             shuffle=False, num_workers=training_config.num_workers,
                             pin_memory=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, mel_augmenter


class FineTuner:
    def __init__(self, model, model_config, training_config):
        self.model = model
        self.cfg = training_config
        self.mcfg = model_config
        self.device = torch.device(training_config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.train_loader, self.val_loader, self.test_loader, self.mel_augmenter = \
            build_loaders(model_config, training_config)
        
        self.logger = TrainingLogger(log_dir=training_config.log_dir, use_wandb=training_config.use_wandb)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Train samples: {len(self.train_loader.dataset)}")

    
    def phase1(self):
        """Train heads with frozen encoder (no precomputation)."""
        print("\n[PHASE 1] Training heads with frozen encoder")
        
        # Set model to use cached embeddings mode (False - use normal forward)
        self.model.use_cached_embeddings = False
        
        # Freeze encoder, train only heads
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["temporal_aggregator", "attention", "query_token", 
                                        "classifier", "domain_classifier", "projection", "ssl_head"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=0.01
        )
        
        # Train directly on data loader (no precomputation)
        for epoch in range(self.cfg.phase1_epochs):
            train_loss = train_epoch(
                self.model, self.train_loader, optimizer, self.criterion,
                device=self.device, gradient_clip_norm=self.cfg.gradient_clip_norm,
                domain="animal"
            )
            
            val_loss, val_acc, _ = validate(
                self.model, self.val_loader, self.criterion,
                device=self.device, domain="animal"
            )
            
            print(f"  Epoch {epoch+1}/{self.cfg.phase1_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        # Save
        torch.save(self.model.state_dict(), f"{self.cfg.checkpoint_dir}/phase1.pt")

    def phase2(self):
        """Train LoRA adapters + heads together."""
        print("\n[PHASE 2] Training LoRA adapters + heads")
        
        # Ensure LoRA params are trainable
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.phase2_lr, weight_decay=self.cfg.weight_decay
        )
        
        self._run_phase(
            phase_id=2, epochs=self.cfg.phase2_epochs,
            optimizer=optimizer, scheduler=None,
            early_stop_path=f"{self.cfg.checkpoint_dir}/phase2.pt",
        )

    def phase3(self):
        """Train domain-specific LoRA weights."""
        print("\n[PHASE 3] Domain-specific LoRA training")
        
        for domain in DOMAINS:
            domain_path = f"{self.cfg.checkpoint_dir}/lora_{domain}.pt"
            print(f"  Training LoRA for {domain}...")
            
            # (Train with domain-specific data, save LoRA weights only)
            self.model.save_lora_weights(domain_path)

    def _run_phase(self, phase_id, epochs, optimizer, scheduler, early_stop_path):
        """Generic training loop."""
        for epoch in range(epochs):
            train_loss = train_epoch(self.model, self.train_loader, optimizer, self.criterion,
                                     device=self.device, gradient_clip_norm=self.cfg.gradient_clip_norm)
            
            val_loss, val_acc, _ = validate(self.model, self.val_loader, self.criterion,
                                            device=self.device)
            
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        torch.save(self.model.state_dict(), early_stop_path)
        print(f"  Saved to {early_stop_path}")


if __name__ == "__main__":
    from models.config import get_configs
    
    model_cfg, train_cfg = get_configs("tiny")
    model = CrossSpeciesVocalizationModel(model_cfg)
    ft = FineTuner(model, model_cfg, train_cfg)
    
    ft.phase1()
    ft.phase2()
    ft.phase3()