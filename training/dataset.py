# training/dataset.py
"""
CrossSpeciesDataset for loading and preprocessing audio data.

Handles ESC-50 and Freesound audio files, converting them to
mel spectrograms compatible with Whisper's expected input format.

Directory structure expected:
    data/raw/
    └── ESC-50/
        └── ESC-50-master/
            ├── audio/          # 2000 .wav files
            └── meta/
                └── esc50.csv   # Metadata with labels
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import warnings
import os

# Set environment variable BEFORE importing torchaudio
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# Now try to import torch and torchaudio
import torch
import torch.nn.functional as F

# Try importing torchaudio, but have a fallback plan
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: torchaudio import failed: {e}")
    print("Will use librosa/soundfile for audio processing")
    TORCHAUDIO_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


class CrossSpeciesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for cross-species audio classification.
    
    Loads audio files and returns:
    - Raw waveform (for MelAugmenter pipeline) OR
    - Pre-computed mel spectrogram (for direct model input)
    
    Supports:
    - ESC-50 structured data (with esc50.csv metadata)
    - Freesound hand-picked files (labels inferred from directory names)
    - Custom mappings for domain-specific classification
    """
    
    # ESC-50 category to our internal label mapping
    ESC50_TO_INTERNAL = {
        # Animal sounds
        "dog": 0,
        "rooster": 5,
        "pig": 0,
        "cow": 0,
        "frog": 9,
        "cat": 3,
        "hen": 5,
        "insects": 5,
        "sheep": 0,
        "crow": 6,
        
        # Human non-verbal sounds
        "coughing": 10,
        "laughing": 13,
        "crying_baby": 12,
        "sneezing": 14,
        "breathing": 14,
        
        # Machinery/Environmental
        "engine": 15,
        "train": 15,
        "airplane": 15,
        "car_horn": 18,
        "chainsaw": 16,
        "siren": 18,
        "church_bells": 18,
        "hand_saw": 16,
        "fireworks": 18,
        "helicopter": 15,
        
        # Other sounds
        "rain": 19,
        "sea_waves": 19,
        "crackling_fire": 17,
        "wind": 19,
        "pouring_water": 19,
        "toilet_flush": 19,
        "thunderstorm": 18,
        "clock_alarm": 18,
        "door_wood_creaks": 17,
        "mouse_click": 17,
        "keyboard_typing": 17,
        "can_opening": 17,
        "washing_machine": 15,
        "vacuum_cleaner": 15,
        "clock_tick": 17,
        "glass_breaking": 17,
        "brushing_teeth": 19,
        "drinking_sipping": 19,
        "footsteps": 14,
        "chirping_birds": 5,
        "water_drops": 19,
    }
    
    def __init__(
        self,
        data_root: str = "data/raw",
        split: str = "train",
        target_duration: float = 10.0,
        sample_rate: int = 16000,
        n_mels: int = 80,
        augment: bool = False,
        augment_prob: float = 0.5,
        mel_augmenter: Optional[object] = None,
        domain_filter: Optional[str] = None,
        return_raw_audio: bool = True,  # If True, returns waveform; if False, returns mel
    ):
        """
        Args:
            data_root: Path to raw data directory
            split: "train", "val", or "test"
            target_duration: Pad/trim all audio to this many seconds
            sample_rate: Target sample rate (Whisper uses 16kHz)
            n_mels: Number of mel filterbanks (Whisper uses 80)
            augment: Whether to apply mel augmentations
            augment_prob: Probability of augmenting each sample
            mel_augmenter: MelAugmenter instance for augmentation
            domain_filter: If set, only return samples from this domain
            return_raw_audio: If True, returns waveform for external mel conversion
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.augment = augment
        self.augment_prob = augment_prob
        self.mel_augmenter = mel_augmenter
        self.domain_filter = domain_filter
        self.return_raw_audio = return_raw_audio
        
        # Target number of samples
        self.target_samples = int(target_duration * sample_rate)
        
        # Initialize audio processing tools
        self._init_audio_processor()
        
        # Load file paths and labels
        self.files, self.labels, self.domains = self._load_data()
        
        # Split indices
        self._create_split()
        
        print(f"Dataset [{split}]: {len(self.indices)} samples"
              + (f" (domain: {domain_filter})" if domain_filter else ""))
    
    def _init_audio_processor(self):
        """Initialize audio processing with proper backend."""
        if TORCHAUDIO_AVAILABLE and not self.return_raw_audio:
            # Create mel spectrogram converter using torchaudio
            self.mel_converter = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=400,
                    hop_length=160,
                    n_mels=80,
                    f_min=0,
                    f_max=8000,
                    power=2.0,
                )
            
            # Convert power spectrogram to dB scale
            self.db_converter = torchaudio.transforms.AmplitudeToDB(
                stype="power",
                top_db=80.0,
            )
        else:
            self.mel_converter = None
            self.db_converter = None
    
    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load file paths and labels from data directory."""
        files = []
        labels = []
        domains = []
        
        # Try both directory structures
        esc50_paths = [
            self.data_root / "ESC-50-master" / "audio",
            self.data_root / "ESC-50" / "ESC-50-master" / "audio",
        ]
        esc50_meta_paths = [
            self.data_root / "ESC-50-master" / "meta" / "esc50.csv",
            self.data_root / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv",
        ]
        
        esc50_audio = None
        esc50_meta = None
        
        for audio_path, meta_path in zip(esc50_paths, esc50_meta_paths):
            if audio_path.exists() and meta_path.exists():
                esc50_audio = audio_path
                esc50_meta = meta_path
                break
        
        if esc50_audio is not None and esc50_meta is not None:
            metadata = pd.read_csv(esc50_meta)
            
            for _, row in metadata.iterrows():
                filename = row["filename"]
                category = row["category"]
                filepath = esc50_audio / filename
                
                if filepath.exists():
                    internal_label = self.ESC50_TO_INTERNAL.get(category)
                    if internal_label is not None:
                        files.append(str(filepath))
                        labels.append(internal_label)
                        domains.append(self._get_domain(internal_label))
        
        # Load Freesound data
        freesound_dir = self.data_root / "freesound"
        if freesound_dir.exists():
            for domain_dir in freesound_dir.iterdir():
                if domain_dir.is_dir():
                    domain_name = domain_dir.name.lower()
                    
                    if "whale" in domain_name or "bird" in domain_name:
                        domain = "animal"
                    elif "human" in domain_name or "cough" in domain_name:
                        domain = "human_nonverbal"
                    elif "machinery" in domain_name or "engine" in domain_name:
                        domain = "machinery"
                    else:
                        domain = "animal"
                    
                    for audio_file in domain_dir.glob("*.wav"):
                        label = self._infer_freesound_label(audio_file, domain)
                        if label is not None:
                            files.append(str(audio_file))
                            labels.append(label)
                            domains.append(domain)
        
        return files, labels, domains
    
    def _infer_freesound_label(self, filepath: Path, domain: str) -> Optional[int]:
        """Infer class label from Freesound filename."""
        name = filepath.stem.lower()
        
        if domain == "animal":
            if any(w in name for w in ["whale", "orca", "humpback", "cetacean"]):
                return 7 if "song" in name or "call" in name else 8
            if any(w in name for w in ["bird", "nightingale", "songbird", "avian"]):
                return 5 if "song" in name else 6
            if any(w in name for w in ["dog", "bark", "canine"]):
                if any(w in name for w in ["growl", "aggressive", "angry"]):
                    return 1
                if any(w in name for w in ["whine", "cry", "distressed"]):
                    return 2
                return 0
            if any(w in name for w in ["cat", "feline", "meow", "kitten"]):
                return 4 if any(w in name for w in ["hiss", "angry", "defensive"]) else 3
            if any(w in name for w in ["frog", "toad", "amphibian"]):
                return 9
            return 0
        
        elif domain == "human_nonverbal":
            if any(w in name for w in ["cough", "coughing"]):
                return 11 if any(w in name for w in ["wet", "phlegm", "chesty"]) else 10
            if any(w in name for w in ["cry", "sob", "weeping"]):
                return 12
            if any(w in name for w in ["laugh", "giggle", "chuckle"]):
                return 13
            if any(w in name for w in ["gasp", "surprise", "shock"]):
                return 14
            return 10
        
        elif domain == "machinery":
            if any(w in name for w in ["engine", "motor"]):
                if any(w in name for w in ["fail", "knock", "bad", "broken"]):
                    return 16
                return 15
            if any(w in name for w in ["bearing", "grind", "scrape"]):
                return 17
            if any(w in name for w in ["alarm", "siren", "emergency", "warning"]):
                return 18
            if any(w in name for w in ["hydraulic", "leak", "hiss", "pneumatic"]):
                return 19
            return 15
        
        return None
    
    def _get_domain(self, label: int) -> str:
        """Get domain string from class label."""
        if label <= 9:
            return "animal"
        elif 10 <= label <= 14:
            return "human_nonverbal"
        else:
            return "machinery"
    
    def _create_split(self):
        """Create train/val/test split indices."""
        n = len(self.files)
        indices = list(range(n))
        
        # Use deterministic random seed
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        
        # 70% train, 15% val, 15% test
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        if self.split == "train":
            self.indices = indices[:train_end]
        elif self.split == "val":
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
        
        # Apply domain filter
        if self.domain_filter:
            self.indices = [
                i for i in self.indices
                if self.domains[i] == self.domain_filter
            ]
    
    def _load_audio(self, filepath: str) -> torch.Tensor:
        """
        Load audio file and preprocess to waveform.
        Uses librosa to avoid torchcodec issues.
        """
        try:
            import librosa
            # Load audio with librosa (much more reliable on Windows)
            audio_np, sr = librosa.load(filepath, sr=None, mono=True)
            
            # Convert to torch tensor [1, samples]
            audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            
            # Resample if needed
            if sr != self.sample_rate:
                audio_np_resampled = librosa.resample(
                    audio_np, orig_sr=sr, target_sr=self.sample_rate
                )
                audio = torch.from_numpy(audio_np_resampled).unsqueeze(0).float()
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return random noise as fallback
            audio = torch.randn(1, self.target_samples)
            return audio
        
        # Pad or trim to target duration
        if audio.shape[1] < self.target_samples:
            padding = self.target_samples - audio.shape[1]
            audio = F.pad(audio, (0, padding))
        elif audio.shape[1] > self.target_samples:
            if self.split == "train":
                start = random.randint(0, audio.shape[1] - self.target_samples)
            else:
                start = (audio.shape[1] - self.target_samples) // 2
            audio = audio[:, start:start + self.target_samples]
        
        return audio
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to log-mel spectrogram."""
        if self.mel_converter is not None:
            # Use torchaudio's mel converter
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            mel_spec = self.mel_converter(audio)  # [1, n_mels, time]
            mel_spec_db = self.db_converter(mel_spec)
            mel_spec_db = (mel_spec_db + 40) / 40  # Rough normalization
            mel_spec_db = mel_spec_db.squeeze(0)  # [n_mels, time]
        else:
            # Fallback: use librosa
            try:
                import librosa
                audio_np = audio.squeeze().numpy()
                
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_np,
                    sr=self.sample_rate,
                    n_fft=400,
                    hop_length=160,
                    n_mels=self.n_mels,
                    fmin=0,
                    fmax=self.sample_rate // 2,
                    power=2.0,
                )
                
                mel_spec_db = librosa.power_to_db(mel_spec, top_db=80.0)
                mel_spec_db = (mel_spec_db + 40) / 40
                mel_spec_db = torch.from_numpy(mel_spec_db).float()
            except Exception as e:
                print(f"Error computing mel spectrogram: {e}")
                # Return random mel spectrogram
                mel_spec_db = torch.randn(self.n_mels, 100).float()
        
        return mel_spec_db
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        actual_idx = self.indices[idx]
        filepath = self.files[actual_idx]
        label = self.labels[actual_idx]
        domain = self.domains[actual_idx]
        
        # Load and preprocess audio
        try:
            audio = self._load_audio(filepath)  # [1, samples]
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return a different sample as fallback
            return self.__getitem__((idx + 1) % len(self.indices))
        
        if self.return_raw_audio:
            # Return raw waveforms for external MelAugmenter pipeline
            return {
                "audio": audio.squeeze(0),  # [samples]
                "mel": None,  # Will be computed by augmenter or collate function
                "label": torch.tensor(label, dtype=torch.long),
                "domain": domain,
                "filepath": filepath,
            }
        else:
            # Convert to mel spectrogram
            mel = self._audio_to_mel(audio)
            
            # Apply mel augmentation during training
            if self.augment and self.mel_augmenter is not None and random.random() < self.augment_prob:
                # Add batch dimension for augmenter
                mel = mel.unsqueeze(0)  # [1, n_mels, time]
                mel, _ = self.mel_augmenter(mel, None)
                mel = mel.squeeze(0)  # [n_mels, time]
            
            return {
                "audio": audio.squeeze(0),  # [samples] - kept for reference
                "mel": mel,  # [n_mels, time]
                "label": torch.tensor(label, dtype=torch.long),
                "domain": domain,
                "filepath": filepath,
            }
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class."""
        from collections import Counter
        label_counts = Counter()
        for idx in self.indices:
            label_counts[self.labels[idx]] += 1
        return dict(label_counts)
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Get count of samples per domain."""
        from collections import Counter
        domain_counts = Counter()
        for idx in self.indices:
            domain_counts[self.domains[idx]] += 1
        return dict(domain_counts)


# ==================================================
# Custom collate function for batching
# ==================================================
def collate_fn(batch, mel_augmenter=None):
    """
    Custom collate function that handles:
    - Mel conversion from audio if needed
    - Augmentation application
    - Padding to same length in batch
    """
    audios = [item["audio"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    domains = [item["domain"] for item in batch]
    filepaths = [item["filepath"] for item in batch]
    
    # If mel is already computed, use it
    if batch[0]["mel"] is not None:
        mels = [item["mel"] for item in batch]
        # Pad mels to same time dimension
        max_time = max(mel.shape[1] for mel in mels)
        padded_mels = []
        for mel in mels:
            if mel.shape[1] < max_time:
                pad_size = max_time - mel.shape[1]
                mel = F.pad(mel, (0, pad_size))
            padded_mels.append(mel)
        mels = torch.stack(padded_mels)  # [B, n_mels, time]
    else:
        # Need to compute mels (this would require the MelAugmenter)
        # For now, return audios and let the model/training loop handle conversion
        mels = None
    
    return {
        "mel": mels,
        "audio": audios,  # List of tensors (variable length)
        "label": labels,
        "domain": domains[0] if len(set(domains)) == 1 else domains,  # Single domain string if all same
        "filepath": filepaths,
    }


# Test the dataset
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)
    
    data_root = "data/raw"
    
    if not Path(data_root).exists():
        print(f"\nData directory not found: {data_root}")
        print("Run data/downloads/download_esc50.py first")
        sys.exit(0)
    
    for split in ["train", "val", "test"]:
        dataset = CrossSpeciesDataset(
            data_root=data_root,
            split=split,
            target_duration=10.0,
            return_raw_audio=True,  # Test raw audio mode
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n{split.upper()} split: {len(dataset)} samples")
            print(f"  Audio shape: {sample['audio'].shape}")
            print(f"  Label: {sample['label']}")
            print(f"  Domain: {sample['domain']}")
            print(f"  File: {Path(sample['filepath']).name}")
            
            dist = dataset.get_class_distribution()
            print(f"  Classes: {len(dist)}")
            
            domain_dist = dataset.get_domain_distribution()
            print(f"  Domains: {domain_dist}")