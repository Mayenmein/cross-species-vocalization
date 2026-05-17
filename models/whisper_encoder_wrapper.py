"""
Wrapper to make WhisperEncoder compatible with PEFT.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


class WhisperEncoderWrapper(nn.Module):
    """
    Wrapper that makes WhisperEncoder accept PEFT's expected input signature.
    
    PEFT expects models to take 'input_ids' but WhisperEncoder takes 'input_features'.
    This wrapper maps between them.
    """
    
    def __init__(self, encoder):
        """
        Args:
            encoder: Whisper encoder instance (from WhisperModel.encoder)
        """
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input_ids=None, input_features=None, attention_mask=None, **kwargs):
        """
        Forward pass that handles both naming conventions.
        
        Args:
            input_ids: PEFT passes this, but we ignore it
            input_features: The actual mel spectrogram Whisper expects
            attention_mask: Optional mask (ignored for encoder-only)
        """
        # Use input_features if provided, otherwise fall back to input_ids
        features = input_features if input_features is not None else input_ids
        
        # Call the original encoder
        return self.encoder(features)
    
    @property
    def device(self):
        return self.encoder.device
    
    @property
    def dtype(self):
        return self.encoder.dtype


def create_peft_encoder(whisper_size: str = "tiny", lora_config: LoraConfig = None):
    """
    Create a PEFT-wrapped Whisper encoder.
    
    Args:
        whisper_size: Size of Whisper model ("tiny", "small", "medium")
        lora_config: PEFT LoraConfig (creates default if None)
    
    Returns:
        PEFT model wrapped around WhisperEncoder
    """
    from transformers import WhisperModel
    
    # Load full model
    model_name = f"openai/whisper-{whisper_size}"
    full_model = WhisperModel.from_pretrained(model_name)
    
    # Extract encoder
    encoder = full_model.encoder
    
    # Wrap encoder for PEFT compatibility
    wrapped_encoder = WhisperEncoderWrapper(encoder)
    
    # Create default LoRA config if not provided
    if lora_config is None:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
    
    # Apply PEFT
    peft_encoder = get_peft_model(wrapped_encoder, lora_config)
    
    # Print info
    trainable = sum(p.numel() for p in peft_encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_encoder.parameters())
    print(f"[PEFT] Encoder trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    return peft_encoder