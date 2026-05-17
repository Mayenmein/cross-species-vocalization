"""
Main training script using PyTorch Lightning
"""
 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from models.config import get_configs, DOMAINS
from training.lightning_module import AudioLightningModule, AudioDataModule


def train():
    # Load configs
    model_config, train_config = get_configs("tiny")
    
    # Setup data module
    data_module = AudioDataModule(train_config, model_config)
    
    # Setup model module
    lightning_model = AudioLightningModule(model_config, train_config)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_config.early_stopping_patience,
            mode="min",
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=train_config.checkpoint_dir,
            filename="best-{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            verbose=True
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
        max_epochs=train_config.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if train_config.use_amp else "32",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=train_config.gradient_clip_norm,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )
    
    # Phase 1: Train heads only
    print("\n" + "="*60)
    print("PHASE 1: Training heads with frozen encoder")
    print("="*60)
    lightning_model.set_phase(1)
    trainer.fit(lightning_model, data_module)
    
    # Save phase 1 checkpoint
    trainer.save_checkpoint(f"{train_config.checkpoint_dir}/phase1.ckpt")
    
    # Phase 2: Train LoRA + heads
    print("\n" + "="*60)
    print("PHASE 2: Training LoRA adapters + heads")
    print("="*60)
    lightning_model.set_phase(2)
    trainer.fit(lightning_model, data_module)
    
    # Save phase 2 checkpoint
    trainer.save_checkpoint(f"{train_config.checkpoint_dir}/phase2.ckpt")
    
    # Phase 3: Domain-specific fine-tuning (optional)
    print("\n" + "="*60)
    print("PHASE 3: Domain-specific training")
    print("="*60)
    lightning_model.set_phase(3)
    
    for domain in DOMAINS:
        print(f"\nTraining on domain: {domain}")
        # Filter dataset for domain (would need custom dataset filtering)
        # For now, just train on all data with domain adapter
        trainer.fit(lightning_model, data_module)
        
        # Save domain-specific LoRA
        save_path = f"{train_config.checkpoint_dir}/lora_{domain}.pt"
        lightning_model.model.save_lora_weights(save_path)
    
    # Final test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    trainer.test(lightning_model, data_module)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()