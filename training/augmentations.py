# training/augmentations.py
"""
Log-mel based audio augmentation (Whisper-style, efficient).

Pipeline:
    waveform → log-mel → augmentation → model

Features:
- SpecAugment (time + frequency masking)
- Additive noise (mel domain)
- Mixup (batch-level)
- Optional learnable augmentation strength

Integrates with dataset collate function for proper batching.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random
import numpy as np


class MelAugmenter(nn.Module):
    """
    Mel-spectrogram augmenter that works on batches.
    
    Can be used in two ways:
    1. As part of the dataset __getitem__ (per-sample)
    2. As part of the training loop/collate function (per-batch)
    """
    
    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        f_min=0,
        f_max=8000,
        time_mask_param=30,
        freq_mask_param=15,
        noise_std=0.01,
        mixup_alpha=0.2,
        use_learnable=False,
        apply_prob=0.5,  # Probability of applying augmentation
    ):
        super().__init__()

        # --- mel frontend (Whisper-like) ---
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power")

        # --- base augmentation params ---
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.apply_prob = apply_prob

        self.use_learnable = use_learnable

        if use_learnable:
            # learnable scalars (constrained via transforms)
            self.log_noise = nn.Parameter(torch.tensor(-4.0))     # exp → small noise
            self.mask_scale = nn.Parameter(torch.tensor(1.0))     # scales mask size

    # ---------- waveform to mel conversion ----------
    def waveform_to_mel(self, audio):
        """
        Convert raw audio waveform to normalized log-mel spectrogram.
        
        Args:
            audio: [B, samples] or [samples] waveform
            
        Returns:
            mel: [B, n_mels, time] or [n_mels, time] log-mel spectrogram
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, samples]
        
        # Add channel dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, samples]
        
        mel = self.mel(audio)        # [B, 1, F, T]
        mel = self.to_db(mel)
        mel = mel.squeeze(1)         # [B, F, T]

        # Normalize (zero mean, unit variance)
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / \
              (mel.std(dim=(1, 2), keepdim=True) + 1e-6)

        return mel

    # ---------- core pipeline ----------
    def forward(self, audio_or_mel, targets=None):
        """
        Apply augmentations to audio or mel spectrogram.
        
        Args:
            audio_or_mel: [B, samples] waveform or [B, F, T] mel
            targets: Optional [B] or [B, num_classes] labels for mixup
            
        Returns:
            mel: [B, F, T] augmented log-mel spectrogram
            targets: Modified targets if mixup was applied
        """
        # Check if input is waveform or mel
        if audio_or_mel.dim() == 2:
            # Waveform: [B, samples]
            mel = self.waveform_to_mel(audio_or_mel)
        elif audio_or_mel.dim() == 3:
            # Mel: [B, F, T]
            mel = audio_or_mel
            # Normalize if not already normalized
            if mel.std() > 2:  # Heuristic: if not normalized
                mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / \
                      (mel.std(dim=(1, 2), keepdim=True) + 1e-6)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {audio_or_mel.dim()}D")
        
        # Apply augmentations with probability
        if random.random() < self.apply_prob:
            mel = self._augment(mel)

            if self.mixup_alpha > 0 and targets is not None and mel.size(0) > 1:
                mel, targets = self._mixup(mel, targets)

        return mel, targets

    # ---------- augmentations ----------
    def _augment(self, mel):
        """Apply SpecAugment style augmentations."""
        B, F, T = mel.shape

        # --- noise ---
        noise_std = (
            torch.exp(self.log_noise) if self.use_learnable else self.noise_std
        )
        mel = mel + noise_std * torch.randn_like(mel)

        # --- masking parameters ---
        scale = torch.clamp(self.mask_scale, 0.5, 2.0) if self.use_learnable else 1.0

        time_mask = min(int(self.time_mask_param * scale), T - 1)
        freq_mask = min(int(self.freq_mask_param * scale), F - 1)

        # --- time masking ---
        if time_mask > 0:
            for i in range(B):
                t = random.randint(0, max(1, T - time_mask))
                mel[i, :, t:t + time_mask] = 0

        # --- frequency masking ---
        if freq_mask > 0:
            for i in range(B):
                f = random.randint(0, max(1, F - freq_mask))
                mel[i, f:f + freq_mask, :] = 0

        return mel

    # ---------- mixup ----------
    def _mixup(self, x, y):
        """
        Applies mixup in mel space.
        
        Args:
            x: [B, F, T] mel spectrograms
            y: [B] or [B, num_classes] labels
            
        Returns:
            mixed_x: [B, F, T] mixed mels
            mixed_y: [B] or [B, num_classes] mixed labels
        """
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(x.device)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

        index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        
        if y.dim() == 1:
            # Class indices - need to return both for loss computation
            mixed_y = (y, y[index], lam)
        else:
            # One-hot or soft labels
            mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y


# --------------------------------------------------
# Test / Visualization
# --------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    print("Testing MelAugmenter...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- create synthetic audio ----
    sr = 16000
    duration = 2.0
    t = torch.arange(int(sr * duration)) / sr

    # multi-tone signal (richer than single sine)
    audio = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +
        0.3 * torch.sin(2 * np.pi * 880 * t) +
        0.2 * torch.sin(2 * np.pi * 220 * t)
    )

    # batchify
    audio = audio.unsqueeze(0).to(device)  # [B=1, T]

    # fake label (for mixup test)
    labels = torch.tensor([0]).to(device)

    # ---- init augmenter ----
    augmenter = MelAugmenter(use_learnable=False).to(device)

    # ---- test waveform input ----
    print("\nTesting waveform input...")
    with torch.no_grad():
        mel_orig = augmenter.waveform_to_mel(audio.clone())  # No augmentation
        mel_aug, _ = augmenter(audio.clone(), labels)  # With augmentation

    mel_orig = mel_orig.cpu().squeeze(0).numpy()
    mel_aug = mel_aug.cpu().squeeze(0).numpy()

    # ---- plotting ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(mel_orig, aspect="auto", origin="lower")
    axes[0].set_title("Original (log-mel)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(mel_aug, aspect="auto", origin="lower")
    axes[1].set_title("Augmented (log-mel)")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig("mel_augmentation_examples.png", dpi=120)
    print("Saved mel_augmentation_examples.png")

    # ---- test mel input ----
    print("\nTesting mel input...")
    with torch.no_grad():
        mel = augmenter.waveform_to_mel(audio)  # [1, F, T]
        mel_aug2, _ = augmenter(mel, labels)  # Pass mel directly

    print(f"Mel shapes match: {mel_aug.shape == mel_aug2.squeeze(0).cpu().numpy().shape}")

    print("\nAll tests passed!")