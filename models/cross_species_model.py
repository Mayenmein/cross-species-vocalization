"""
Cross-Species Vocalization Model (Multi-Objective Learning)

Objectives:
- Supervised classification (sound type)
- Domain adversarial learning (GRL)
- Contrastive cross-species representation learning
- Self-supervised feature reconstruction

Architecture:
Whisper → Domain Adapter → Temporal Encoder → Pooling → Shared Representation
                                                     ├── Class Head
                                                     ├── Domain Head (GRL)
                                                     ├── Projection Head (Contrastive)
                                                     └── SSL Head (Reconstruction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper 

from models.config import ModelConfig


# =========================================================
# Gradient Reversal Layer (Domain Adversarial Learning)
# =========================================================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.λ * grad_output, None


def grad_reverse(x, λ=1.0):
    return GRL.apply(x, λ)


# =========================================================
# Domain Adapter (lightweight residual)
# =========================================================
class DomainAdapter(nn.Module):
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


# =========================================================
# Projection Head (Contrastive Learning)
# =========================================================
class ProjectionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# =========================================================
# Main Model
# =========================================================
class CrossSpeciesVocalizationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # -------------------------
        # Whisper Encoder (frozen initially)
        # -------------------------
        self.whisper = whisper.load_model(config.whisper_size)
        self.encoder = self.whisper.encoder
        self._freeze_encoder()

        # -------------------------
        # Domain Adapters
        # -------------------------
        self.use_domain_adapters = config.use_domain_adapters
        self.domain_adapters = nn.ModuleDict({
            d: DomainAdapter(config.encoder_dim)
            for d in ["animal", "human_nonverbal", "machinery"]
        })

        # -------------------------
        # Temporal Encoder (GRU)
        # -------------------------
        self.temporal_aggregator = nn.GRU(
            input_size=config.encoder_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=config.lstm_bidirectional,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0
        )

        self.temporal_dim = config.lstm_output_dim

        # -------------------------
        # Attention pooling
        # -------------------------
        self.attention = nn.Sequential(
            nn.Linear(self.temporal_dim, self.temporal_dim // 2),
            nn.Tanh(),
            nn.Linear(self.temporal_dim // 2, 1)
        )

        # -------------------------
        # Query token for class-specific attention
        # -------------------------
        self.query_token = nn.Parameter(torch.randn(1, 1, self.temporal_dim))

        # -------------------------
        # Shared representation heads
        # -------------------------
        self.projection = ProjectionHead(self.temporal_dim)

        self.ssl_head = nn.Linear(self.temporal_dim, self.temporal_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_dim, 256),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(256, config.num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.temporal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.num_domains)
        )

    # =========================================================
    # Forward Pass
    # =========================================================
    def forward(self, x, domain="animal", grl_lambda=1.0, return_ssl_target=False):
        # x is raw audio [B, samples]
        mel = whisper.log_mel_spectrogram(x)  # [B, 80, T_actual]
        
        # Pad to Whisper's expected length (3000 for 30 seconds)
        expected_len = 3000  # Whisper's n_audio_ctx for 30 seconds
        
        if mel.shape[2] < expected_len:
            mel = F.pad(mel, (0, expected_len - mel.shape[2]))
        elif mel.shape[2] > expected_len:
            mel = mel[:, :, :expected_len]
        
        x = self.encoder(mel)  # [B, T, D]

        # -------------------------
        # 2. Domain adaptation
        # -------------------------
        if self.use_domain_adapters:
            adapter = self.domain_adapters[domain] if domain in self.domain_adapters else self.domain_adapters["animal"]
            x = adapter(x)

        # -------------------------
        # 3. Temporal encoding
        # -------------------------
        x, _ = self.temporal_aggregator(x)  # [B, T, temporal_dim]

        # -------------------------
        # 4. Attention pooling
        # -------------------------
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        z = (x * attn_weights).sum(dim=1)  # [B, temporal_dim]

        # -------------------------
        # 5. Heads
        # -------------------------
        logits = self.classifier(z)

        domain_logits = self.domain_classifier(grad_reverse(z, grl_lambda))

        z_proj = self.projection(z)

        z_ssl = self.ssl_head(z)

        out = {
            "logits": logits,
            "domain_logits": domain_logits,
            "projection": z_proj,
            "features": z,
            "ssl_pred": z_ssl
        }

        if return_ssl_target:
            out["ssl_target"] = z.detach()

        return out

    # =========================================================
    # Freezing utilities
    # =========================================================
    def _freeze_encoder(self):
        """Freeze all Whisper encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _freeze_whisper(self):
        """Alias for _freeze_encoder for compatibility."""
        self._freeze_encoder()

    def _unfreeze_top_layers(self, n: int = 2):
        """Unfreeze the top n layers of Whisper encoder."""
        blocks = self.encoder.blocks
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True

    def unfreeze_top_layers(self, n: int = 2):
        """Public alias for _unfreeze_top_layers."""
        self._unfreeze_top_layers(n)

    # =========================================================
    # Inference
    # =========================================================
    def predict(self, x: torch.Tensor, domain: str = "animal"):
        self.eval()
        with torch.no_grad():
            out = self(x, domain=domain)
            probs = F.softmax(out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)

        return {
            "class_id": pred.item(),
            "confidence": conf.item(),
            "embedding": out["features"]
        }

    def count_params(self, trainable=False):
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable)
        )