"""
PyTorch Lightning Module for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, AUROC
from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import ModelConfig, TrainingConfig, DOMAIN_MAP
import wandb

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
        
    def forward(self, x, domain="animal"):
        return self.model(x, domain=domain)
    
    def training_step(self, batch, batch_idx):
        # Process batch
        x = self._prepare_audio(batch["audio"])
        y = batch["label"]
        
        # Get domain for this batch - handle both string and list
        domain = batch.get("domain", "animal")
        if isinstance(domain, list):
            # If mixed domains, use first domain (batches should be single domain after fix)
            # Or implement per-sample domain handling
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
        
        # SSL loss (reconstruction) - simplified, removed to reduce complexity
        # Keeping only if needed
        
        # Log metrics
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # batch["audio"] is a list of tensors; stack them into a unified tensor
        x_raw = torch.stack(batch["audio"]).to(self.device)
        y = batch["label"]
        domains = batch["domain"]
        
        # Handle both single string (fallback) and list of domains safely
        if isinstance(domains, str):
            domains = [domains] * len(y)
            
        # Group batch indices by their respective domains to route them through correct adapters
        from collections import defaultdict
        domain_to_indices = defaultdict(list)
        for idx, d in enumerate(domains):
            domain_to_indices[d].append(idx)
            
        # Destination tensor initializes as Float32 by default
        all_logits = torch.zeros((len(y), self.model_config.num_classes), device=self.device)
        
        # Route samples slice-by-slice based on their domain
        for domain_name, indices in domain_to_indices.items():
            sub_x = x_raw[indices]
            
            # Pass through your existing audio prep logic
            x_processed = self._prepare_audio(sub_x) 
            
            # Forward pass output is intercepted by AMP and returned as Float16 (Half)
            out = self.model(x_processed, domain=domain_name)
            
            # FIX (Option B): Upcast the Half source tensor to Float32 using .float() 
            # before saving it into the Float32 destination slice
            all_logits[indices] = out["logits"].float()
            
        # Compute the standard loss on the complete, uncorrupted batch
        loss = F.cross_entropy(all_logits, y)
        
        # Update tracking metrics accurately
        self.val_acc(all_logits, y)
        self.val_f1(all_logits, y)
        self.val_auroc(all_logits, y)
        
        # Keep track of predictions for the final confusion matrix evaluation
        if batch_idx == 0:
            self.val_preds = all_logits.argmax(dim=-1)
            self.val_targets = y
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log confusion matrix to WandB as an image."""
        if hasattr(self, 'val_preds') and hasattr(self, 'val_targets'):
            # Update confusion matrix state
            self.val_confusion.update(self.val_preds, self.val_targets)
            
            # Get confusion matrix as tensor
            cm = self.val_confusion.compute()
            
            # Only log to WandB (not as scalar)
            if self.logger is not None and hasattr(self.logger, 'experiment'):
                # Convert to numpy for plotting
                cm_np = cm.cpu().numpy()
                
                # Create figure
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    cm_np,
                    annot=False,  # Too many classes for annotations
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    cbar=True
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title('Confusion Matrix - Validation')
                
                # Log to WandB
                self.logger.experiment.log({
                    "val_confusion_matrix": wandb.Image(fig)
                })
                plt.close(fig)
            
            # Reset for next epoch
            self.val_confusion.reset()
    
    def test_step(self, batch, batch_idx):
        x_raw = torch.stack(batch["audio"]).to(self.device)
        y = batch["label"]
        domains = batch["domain"]
        
        if isinstance(domains, str):
            domains = [domains] * len(y)
            
        from collections import defaultdict
        domain_to_indices = defaultdict(list)
        for idx, d in enumerate(domains):
            domain_to_indices[d].append(idx)
            
        all_logits = torch.zeros((len(y), self.model_config.num_classes), device=self.device)
        
        for domain_name, indices in domain_to_indices.items():
            sub_x = x_raw[indices]
            x_processed = self._prepare_audio(sub_x)
            out = self.model(x_processed, domain=domain_name)
            
            # FIX (Option B): Upcast the model output slice to float to match destination precision
            all_logits[indices] = out["logits"].float()
            
        loss = F.cross_entropy(all_logits, y)
        self.test_acc(all_logits, y)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
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
            for domain_idx, (domain, classes) in enumerate(DOMAIN_MAP.items()):
                if label.item() in classes:
                    domain_labels.append(domain_idx)
                    break
        return torch.tensor(domain_labels, device=class_labels.device)
    
    def configure_optimizers(self):
        # Select parameters based on the active phase
        if self.current_phase == 1:
            # Only linear classifier heads and domain adapters
            params = [p for p in self.parameters() if p.requires_grad]
            lr = self.train_config.phase1_lr
            epochs = self.train_config.phase1_epochs
        elif self.current_phase == 2:
            params = [p for p in self.parameters() if p.requires_grad]
            lr = self.train_config.phase2_lr
            epochs = self.train_config.phase2_epochs
        else:
            params = [p for p in self.parameters() if p.requires_grad]
            lr = self.train_config.phase3_lr
            epochs = self.train_config.phase3_epochs

        optimizer = torch.optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=self.train_config.weight_decay
        )
        
        # CRITICAL FIX: Implement Linear Warmup + Cosine Annealing Scheduler
        # Warm up for 3 epochs, then smoothly decay to 0 over the remaining phase epochs
        warmup_epochs = 3
        
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Linear scale up from 10% to 100% of the target LR
                return 0.1 + 0.9 * (current_epoch / warmup_epochs)
            else:
                # Cosine decay down to 1% of the target LR
                progress = (current_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                import math
                return 0.01 + 0.99 * (0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def set_phase(self, phase: int):
        """Set training phase and freeze/unfreeze accordingly."""
        self.current_phase = phase
        
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        if phase == 1:
            # Phase 1: Train only heads
            trainable_keywords = ["classifier", "projection", "ssl_head", "domain_classifier"]
            print("[Phase 1] Training heads only (classifier, projection, SSL, domain)")
            
            for name, param in self.model.named_parameters():
                if any(k in name for k in trainable_keywords):
                    param.requires_grad = True
                    
        elif phase == 2:
            # Phase 2: Train heads + LoRA adapters
            trainable_keywords = ["classifier", "projection", "ssl_head", "domain_classifier", "lora_"]
            print("[Phase 2] Training LoRA adapters + heads")
            
            for name, param in self.model.named_parameters():
                if any(k in name for k in trainable_keywords):
                    param.requires_grad = True
                    
        else:
            # Phase 3: Train everything
            print("[Phase 3] Full fine-tuning (all parameters)")
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Print summary
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, train_config: TrainingConfig, model_config: ModelConfig):
        super().__init__()
        self.train_config = train_config
        self.model_config = model_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mel_augmenter = None
    
    def setup(self, stage=None):
        from training.dataset import CrossSpeciesDataset, collate_fn
        from training.augmentations import MelAugmenter
        
        # Setup augmenter
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
        from training.dataset import collate_fn
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        from training.dataset import collate_fn
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def test_dataloader(self):
        from torch.utils.data import DataLoader
        from training.dataset import collate_fn
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )