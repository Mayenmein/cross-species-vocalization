"""
PyTorch Lightning Module for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, AUROC
from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import ModelConfig, TrainingConfig, DOMAINS


class AudioLightningModule(pl.LightningModule):
    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_config = model_config
        self.train_config = train_config
        self.model = CrossSpeciesVocalizationModel(model_config)
        
        # TorchMetrics
        self.train_acc = Accuracy(task="multiclass", num_classes=model_config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=model_config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=model_config.num_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=model_config.num_classes, average="weighted")
        self.val_auroc = AUROC(task="multiclass", num_classes=model_config.num_classes)
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=model_config.num_classes)
        
        # Domain metrics
        self.domain_acc = Accuracy(task="multiclass", num_classes=model_config.num_domains)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Track current phase
        self.current_phase = 1  # 1, 2, or 3
        self.current_domain = "animal"
        
    def forward(self, x, domain="animal"):
        return self.model(x, domain=domain)
    
    def training_step(self, batch, batch_idx):
        # Process batch
        x = self._prepare_audio(batch["audio"])
        y = batch["label"]
        
        # Get domain from batch (if available and phase 3)
        domain = batch.get("domain", "animal")
        if isinstance(domain, list):
            domain = domain[0]
        
        # Forward pass
        out = self.model(x, domain=domain)
        logits = out["logits"]
        loss = self.criterion(logits, y)
        
        # Domain confusion loss (phase 2+)
        if self.current_phase >= 2 and self.model_config.use_domain_confusion:
            domain_logits = out["domain_logits"]
            domain_labels = self._get_domain_labels(y, x.size(0))
            domain_loss = self.criterion(domain_logits, domain_labels)
            loss = loss + self.model_config.domain_confusion_weight * domain_loss
        
        # SSL loss (reconstruction)
        if self.current_phase >= 2:
            ssl_loss = F.mse_loss(out["ssl_pred"], out["features"].detach())
            loss = loss + 0.1 * ssl_loss
        
        # Log metrics
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = self._prepare_audio(batch["audio"])
        y = batch["label"]
        domain = batch.get("domain", "animal")
        if isinstance(domain, list):
            domain = domain[0]
        
        out = self.model(x, domain=domain)
        logits = out["logits"]
        loss = self.criterion(logits, y)
        
        # Update metrics
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_auroc(logits, y)
        
        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_auroc", self.val_auroc)
        
        # Store predictions for confusion matrix
        if batch_idx == 0:
            self.val_preds = logits.argmax(dim=-1)
            self.val_targets = y
    
    def on_validation_epoch_end(self):
        # Log confusion matrix
        if hasattr(self, 'val_preds'):
            self.val_confusion.update(self.val_preds, self.val_targets)
            self.log("val_confusion", self.val_confusion)
            self.val_confusion.reset()
    
    def test_step(self, batch, batch_idx):
        x = self._prepare_audio(batch["audio"])
        y = batch["label"]
        domain = batch.get("domain", "animal")
        if isinstance(domain, list):
            domain = domain[0]
        
        out = self.model(x, domain=domain)
        logits = out["logits"]
        loss = self.criterion(logits, y)
        
        self.test_acc(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, prog_bar=True)
    
    def _prepare_audio(self, audio_list):
        """Pad audio batch."""
        max_len = max(a.shape[0] for a in audio_list)
        padded = torch.stack([
            F.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a
            for a in audio_list
        ])
        return padded.to(self.device)
    
    def _get_domain_labels(self, class_labels, batch_size):
        """Get domain labels from class labels."""
        domain_labels = []
        for label in class_labels:
            for domain_idx, (domain, classes) in enumerate(self.model_config.DOMAIN_MAP.items()):
                if label.item() in classes:
                    domain_labels.append(domain_idx)
                    break
        return torch.tensor(domain_labels, device=class_labels.device)
    
    def configure_optimizers(self):
        # Phase-specific optimizer configuration
        if self.current_phase == 1:
            # Train only heads
            trainable_params = []
            for name, param in self.model.named_parameters():
                if any(k in name for k in ["temporal_aggregator", "attention", "classifier", 
                                           "domain_classifier", "projection", "ssl_head"]):
                    trainable_params.append(param)
            
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.train_config.phase1_lr,
                weight_decay=self.train_config.weight_decay
            )
        elif self.current_phase == 2:
            # Train LoRA + heads
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.train_config.phase2_learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        else:
            # Phase 3: Domain-specific
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.train_config.phase3_learning_rate,
                weight_decay=self.train_config.weight_decay
            )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
    
    def set_phase(self, phase: int):
        """Set training phase and freeze/unfreeze accordingly."""
        self.current_phase = phase
        
        # Define trainable keywords for each phase
        if phase == 1:
            # Phase 1: Train only heads (classifier, projection, SSL, domain)
            trainable_keywords = [
                "classifier",
                "projection", 
                "ssl_head",
                "domain_classifier"
            ]
            print("[Phase 1] Training heads only (classifier, projection, SSL, domain)")
            
        elif phase == 2:
            # Phase 2: Train heads + LoRA adapters
            trainable_keywords = [
                "classifier",
                "projection",
                "ssl_head", 
                "domain_classifier",
                "lora_"  # LoRA parameters
            ]
            print("[Phase 2] Training LoRA adapters + heads")
            
        else:
            # Phase 3: Train everything (full fine-tuning)
            trainable_keywords = None  # None means all parameters trainable
            print("[Phase 3] Full fine-tuning (all parameters)")
        
        # Apply freezing/unfreezing
        for name, param in self.model.named_parameters():
            if phase == 3:
                # Phase 3: Unfreeze everything
                param.requires_grad = True
            else:
                # Phases 1 and 2: Only unfreeze matching parameters
                is_trainable = any(keyword in name for keyword in trainable_keywords)
                param.requires_grad = is_trainable
        
        # Print summary of trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, train_config: TrainingConfig, model_config: ModelConfig):
        super().__init__()
        self.train_config = train_config
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        from training.dataset import CrossSpeciesDataset
        from training.augmentations import MelAugmenter
        
        # Setup augmenter
        self.mel_augmenter = None
        if self.train_config.use_augmentation:
            self.mel_augmenter = MelAugmenter(
                sample_rate=self.model_config.sample_rate,
                n_mels=self.model_config.n_mels,
                apply_prob=self.train_config.augmentation_prob,
            )
        
        # Create datasets
        self.train_dataset = CrossSpeciesDataset(
            data_root=self.train_config.data_root,
            split="train",
            target_duration=self.model_config.target_duration,
            sample_rate=self.model_config.sample_rate,
            n_mels=self.model_config.n_mels,
            augment=self.train_config.use_augmentation,
            augment_prob=self.train_config.augmentation_prob,
            mel_augmenter=self.mel_augmenter,
            return_raw_audio=True,
        )
        
        self.val_dataset = CrossSpeciesDataset(
            data_root=self.train_config.data_root,
            split="val",
            target_duration=self.model_config.target_duration,
            sample_rate=self.model_config.sample_rate,
            n_mels=self.model_config.n_mels,
            augment=False,
            return_raw_audio=True,
        )
        
        self.test_dataset = CrossSpeciesDataset(
            data_root=self.train_config.data_root,
            split="test",
            target_duration=self.model_config.target_duration,
            sample_rate=self.model_config.sample_rate,
            n_mels=self.model_config.n_mels,
            augment=False,
            return_raw_audio=True,
        )
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size * 2,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_config.batch_size * 2,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )