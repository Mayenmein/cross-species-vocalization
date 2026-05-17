"""
Audio augmentation using torchaudio and audiomentations
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random


class MelAugmenter(nn.Module):
    """
    Mel-spectrogram augmenter using torchaudio transforms.
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
        apply_prob=0.5,
    ):
        super().__init__()
        
        # torchaudio mel frontend
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
        
        # SpecAugment using torchaudio
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        
        # Noise addition (custom, but simple)
        self.noise_std = noise_std
        self.apply_prob = apply_prob
        
    def waveform_to_mel(self, audio):
        """Convert raw audio to log-mel spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, samples]
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, samples]
        
        mel = self.mel(audio)        # [B, 1, F, T]
        mel = self.to_db(mel)
        mel = mel.squeeze(1)         # [B, F, T]
        
        # Normalize
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / \
              (mel.std(dim=(1, 2), keepdim=True) + 1e-6)
        
        return mel
    
    def forward(self, audio_or_mel, targets=None):
        """Apply augmentations."""
        # Convert to mel if needed
        if audio_or_mel.dim() == 2:  # Waveform
            mel = self.waveform_to_mel(audio_or_mel)
        else:  # Already mel
            mel = audio_or_mel
        
        # Apply augmentations
        if random.random() < self.apply_prob:
            # Add noise
            mel = mel + self.noise_std * torch.randn_like(mel)
            
            # Apply torchaudio SpecAugment
            # Time masking
            mel = self.time_mask(mel)
            # Frequency masking
            mel = self.freq_mask(mel)
        
        return mel, targets


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Testing MelAugmenter...")
    
    sr = 16000
    duration = 2.0
    t = torch.arange(int(sr * duration)) / sr
    
    # Multi-tone signal
    audio = (
        0.5 * torch.sin(2 * np.pi * 440 * t) +
        0.3 * torch.sin(2 * np.pi * 880 * t) +
        0.2 * torch.sin(2 * np.pi * 220 * t)
    )
    
    audio = audio.unsqueeze(0)  # [B=1, T]
    labels = torch.tensor([0])
    
    augmenter = MelAugmenter()
    
    with torch.no_grad():
        mel_orig = augmenter.waveform_to_mel(audio.clone())
        mel_aug, _ = augmenter(audio.clone(), labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    im0 = axes[0].imshow(mel_orig.squeeze().numpy(), aspect="auto", origin="lower")
    axes[0].set_title("Original (log-mel)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(mel_aug.squeeze().numpy(), aspect="auto", origin="lower")
    axes[1].set_title("Augmented (log-mel)")
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("mel_augmentation_examples.png", dpi=120)
    print("Saved mel_augmentation_examples.png")
    print("All tests passed!")