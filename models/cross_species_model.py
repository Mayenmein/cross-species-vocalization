"""
Cross-Species Vocalization Model with PEFT LoRA on Encoder Only.
Simplified architecture: Encoder → Pooling → Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from models.config import ModelConfig, DOMAINS
from models.whisper_encoder_wrapper import create_peft_encoder


class DomainAdapter(nn.Module):
    """Lightweight adapter applied after pooling (not on raw sequence)."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.net(x)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation."""
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.λ * grad_output, None


def grad_reverse(x, λ=1.0):
    return GRL.apply(x, λ)


class CrossSpeciesVocalizationModel(nn.Module):
    """
    Simplified model: Whisper Encoder → Pooling → Domain Adapter → Classifier
    
    Removed redundant GRU and attention pooling over time.
    Uses simple mean pooling over time dimension.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── PEFT-Wrapped Whisper Encoder (encoder only) ──
        print(f"[Encoder] Creating PEFT-wrapped Whisper-{config.whisper_size} encoder...")
        self.encoder = create_peft_encoder(
            whisper_size=config.whisper_size,
            lora_config=None
        )
        
        # Print trainable parameters
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        print(f"[PEFT] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        # ── Domain Adapters (applied AFTER pooling) ──
        self.domain_adapters = nn.ModuleDict({
            d: DomainAdapter(config.encoder_dim)
            for d in DOMAINS
        })

        # ── Classifier Head (simplified) ──
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.encoder_dim, config.classifier_hidden[0]),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden[0], config.classifier_hidden[1]),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden[1], config.num_classes)
        )
        
        # ── Domain Classifier (for domain confusion) ──
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_domains)
        )
        
        # ── Auxiliary heads ──
        self.projection = ProjectionHead(config.encoder_dim)
        self.ssl_head = nn.Linear(config.encoder_dim, config.encoder_dim)

    def forward(self, x, domain="animal", grl_lambda=1.0, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Raw audio waveform [batch, samples]
            domain: Domain hint for adapter selection
            grl_lambda: Gradient reversal strength
            return_attention: Kept for API compatibility (ignored)
        """
        # Convert raw audio to log-mel spectrogram
        mel = whisper.log_mel_spectrogram(x)
        expected_len = 3000
        if mel.shape[2] < expected_len:
            mel = F.pad(mel, (0, expected_len - mel.shape[2]))
        elif mel.shape[2] > expected_len:
            mel = mel[:, :, :expected_len]
        
        # Encode with Whisper encoder
        outputs = self.encoder(input_features=mel)
        
        # Extract hidden states
        if hasattr(outputs, 'last_hidden_state'):
            encoded = outputs.last_hidden_state  # [batch, time, dim]
        else:
            encoded = outputs
        
        # Mean pooling over time dimension (simplest and most effective)
        pooled = encoded.mean(dim=1)  # [batch, dim]
        
        # Apply domain adapter after pooling
        pooled = self.domain_adapters[domain](pooled)
        
        # Classification
        out = {
            "logits": self.classifier(pooled),
            "domain_logits": self.domain_classifier(grad_reverse(pooled, grl_lambda)),
            "projection": self.projection(pooled),
            "features": pooled,
            "ssl_pred": self.ssl_head(pooled),
        }
        
        return out

    def predict(self, x, domain="animal"):
        """Single sample prediction."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x, domain=domain)
            probs = F.softmax(out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
        return {
            "class_id": pred.item(), 
            "confidence": conf.item(), 
            "embedding": out["features"].cpu().numpy()
        }

    def count_params(self, trainable=False):
        """Count parameters."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_lora_weights(self, path):
        """Save PEFT LoRA weights."""
        self.encoder.save_pretrained(path)
        print(f"[LoRA] Saved to {path}")

    def load_lora_weights(self, path):
        """Load PEFT LoRA weights."""
        from peft import PeftModel
        base_encoder = self.encoder.base_model.model.encoder
        self.encoder = PeftModel.from_pretrained(base_encoder, path)
        print(f"[LoRA] Loaded from {path}")


if __name__ == "__main__":
    from models.config import ModelConfig
    
    config = ModelConfig(whisper_size="tiny")
    model = CrossSpeciesVocalizationModel(config)
    
    print(f"Total params: {model.count_params():,}")
    print(f"Trainable: {model.count_params(trainable=True):,}")
    
    dummy = torch.randn(2, 16000 * 10)
    with torch.no_grad():
        out = model(dummy, domain="animal")
    print(f"Logits: {out['logits'].shape}")
    print("✓ Model works")