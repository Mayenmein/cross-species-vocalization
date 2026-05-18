"""
Main training script using PyTorch Lightning with proper domain handling.
"""
 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path

from models.config import get_configs, DOMAINS
from training.lightning_module import AudioLightningModule, AudioDataModule 

def train():
    # Load configs
    model_config, train_config = get_configs("tiny")
    
    # Create output directories
    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    data_module = AudioDataModule(train_config, model_config)
    data_module.setup()
    
    # Setup model module
    lightning_model = AudioLightningModule(model_config, train_config)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            patience=train_config.early_stopping_patience,
            mode="min", 
            monitor="val_loss",
        ),
        ModelCheckpoint(
            dirpath=train_config.checkpoint_dir,
            filename="best-{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=train_config.checkpoint_dir,
            filename="last",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]
    
    # Logger
    if train_config.use_wandb:
        logger = WandbLogger(
            project="cross-species-vocalization",
            name=f"whisper-{model_config.whisper_size}",
            log_model=True,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=train_config.log_dir,
            name="cross_species",
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config.phase1_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if train_config.use_amp else "32",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=train_config.gradient_clip_norm,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )
    
    # =========================================================
    # PHASE 1: Train heads with frozen encoder
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training heads with frozen encoder")
    print("=" * 60)
    
    lightning_model.set_phase(1)
    trainer.fit(lightning_model, data_module)
    
    # Save phase 1 checkpoint
    trainer.save_checkpoint(f"{train_config.checkpoint_dir}/phase1.ckpt")
    
    # =========================================================
    # PHASE 2: Train LoRA adapters + heads
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Training LoRA adapters + heads")
    print("=" * 60)
    
    # Reset trainer for phase 2
    trainer.max_epochs = train_config.phase2_epochs
    lightning_model.set_phase(2)
    trainer.fit(lightning_model, data_module)
    
    # Save phase 2 checkpoint
    trainer.save_checkpoint(f"{train_config.checkpoint_dir}/phase2.ckpt")
    
    # =========================================================
    # PHASE 3: Domain-specific fine-tuning
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Domain-specific fine-tuning")
    print("=" * 60)
    
    for domain in DOMAINS:
        print(f"\nTraining on domain: {domain}")
        
        # Create domain-specific data module
        domain_data_module = AudioDataModule(train_config, model_config)
        # Override dataset to filter by domain
        domain_data_module.train_dataset.domain_filter = domain
        domain_data_module.val_dataset.domain_filter = domain
        domain_data_module.setup()
        
        lightning_model.set_phase(3)
        trainer.max_epochs = train_config.phase3_epochs
        trainer.fit(lightning_model, domain_data_module)
        
        # Save domain-specific LoRA weights
        save_path = f"{train_config.checkpoint_dir}/lora_{domain}.pt"
        lightning_model.model.save_lora_weights(save_path)
    
    # =========================================================
    # FINAL EVALUATION
    # =========================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    trainer.test(lightning_model, data_module)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Checkpoints saved to: {train_config.checkpoint_dir}")
    print(f"Logs saved to: {train_config.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()