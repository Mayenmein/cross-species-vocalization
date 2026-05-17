"""
CrossSpeciesDataset with sklearn train/test split
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import warnings
import os
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

import torchaudio 
warnings.filterwarnings("ignore", category=UserWarning)


class CrossSpeciesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for cross-species audio classification.
    """
    
    ESC50_TO_INTERNAL = {
        # Animal
        "dog": 0, "rooster": 5, "pig": 0, "cow": 0, "frog": 9,
        "cat": 3, "hen": 5, "insects": 5, "sheep": 0, "crow": 6,
        "chirping_birds": 5,
        # Human Health
        "coughing": 10, "sneezing": 11, "breathing": 12,
        "crying_baby": 13, "laughing": 16, "footsteps": 17,
        # Machinery
        "engine": 18, "train": 18, "airplane": 18, "helicopter": 18,
        "chainsaw": 19, "hand_saw": 21, "car_horn": 23, "siren": 23,
        "clock_alarm": 23, "church_bells": 23, "fireworks": 22,
        # Environmental
        "rain": 25, "wind": 26, "thunderstorm": 27, "sea_waves": 28,
        "pouring_water": 28, "water_drops": 28, "toilet_flush": 28,
        "crackling_fire": 29, "washing_machine": 28, "brushing_teeth": 28,
        "drinking_sipping": 28,
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
        return_raw_audio: bool = True,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
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
        
        self.target_samples = int(target_duration * sample_rate)
        
        # Load all data
        self.files, self.labels, self.domains = self._load_data()
        
        # Create splits using sklearn
        self._create_splits(train_split, val_split, test_split)
        
        # Apply domain filter
        if self.domain_filter:
            self.indices = [
                i for i in self.indices
                if self.domains[i] == self.domain_filter
            ]
        
        print(f"Dataset [{split}]: {len(self.indices)} samples"
              + (f" (domain: {domain_filter})" if domain_filter else ""))
    
    def _load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load file paths and labels."""
        files, labels, domains = [], [], []
        
        # Load ESC-50
        esc50_paths = [
            self.data_root / "ESC-50-master" / "audio",
            self.data_root / "ESC-50" / "ESC-50-master" / "audio",
        ]
        esc50_meta_paths = [
            self.data_root / "ESC-50-master" / "meta" / "esc50.csv",
            self.data_root / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv",
        ]
        
        for audio_path, meta_path in zip(esc50_paths, esc50_meta_paths):
            if audio_path.exists() and meta_path.exists():
                metadata = pd.read_csv(meta_path)
                for _, row in metadata.iterrows():
                    filepath = audio_path / row["filename"]
                    if filepath.exists():
                        internal_label = self.ESC50_TO_INTERNAL.get(row["category"])
                        if internal_label is not None:
                            files.append(str(filepath))
                            labels.append(internal_label)
                            domains.append(self._get_domain(internal_label))
                break
        
        # Load Freesound (if exists)
        freesound_dir = self.data_root / "freesound"
        if freesound_dir.exists():
            for domain_dir in freesound_dir.iterdir():
                if domain_dir.is_dir():
                    domain_name = domain_dir.name.lower()
                    domain = self._infer_domain_from_dir(domain_name)
                    for audio_file in domain_dir.glob("*.wav"):
                        label = self._infer_freesound_label(audio_file, domain)
                        if label is not None:
                            files.append(str(audio_file))
                            labels.append(label)
                            domains.append(domain)
        
        return files, labels, domains
    
    def _infer_domain_from_dir(self, dirname: str) -> str:
        """Infer domain from directory name."""
        if any(w in dirname for w in ["whale", "bird", "animal", "dog", "cat", "frog"]):
            return "animal"
        elif any(w in dirname for w in ["human", "cough", "cry", "laugh", "health"]):
            return "human_health"
        elif any(w in dirname for w in ["machinery", "engine", "alarm", "industrial"]):
            return "machinery"
        else:
            return "environmental"
    
    def _infer_freesound_label(self, filepath: Path, domain: str) -> Optional[int]:
        """Infer class label from filename."""
        name = filepath.stem.lower()
        
        if domain == "animal":
            if any(w in name for w in ["whale", "orca", "humpback"]):
                return 7 if "song" in name else 8
            if any(w in name for w in ["bird", "nightingale"]):
                return 5 if "song" in name else 6
            if any(w in name for w in ["dog", "bark"]):
                return 1 if "growl" in name else 2 if "whine" in name else 0
            if any(w in name for w in ["cat", "meow"]):
                return 4 if "hiss" in name else 3
            if any(w in name for w in ["frog"]):
                return 9
            return 0
        
        elif domain == "human_health":
            if "cough" in name:
                return 11 if "wet" in name else 10
            if any(w in name for w in ["cry", "sob"]):
                return 13
            if "laugh" in name:
                return 16
            if "gasp" in name:
                return 17
            return 10
        
        elif domain == "machinery":
            if "engine" in name:
                return 19 if any(w in name for w in ["fail", "knock"]) else 18
            if "bearing" in name:
                return 20
            if any(w in name for w in ["alarm", "siren"]):
                return 23
            return 18
        
        else:  # environmental
            if "rain" in name:
                return 25
            if "wind" in name:
                return 26
            if "thunder" in name:
                return 27
            if any(w in name for w in ["water", "river", "stream"]):
                return 28
            if "fire" in name:
                return 29
            return 25
    
    def _get_domain(self, label: int) -> str:
        """Get domain from class label."""
        if label <= 9:
            return "animal"
        elif 10 <= label <= 17:
            return "human_health"
        elif 18 <= label <= 24:
            return "machinery"
        else:
            return "environmental"
    
    def _create_splits(self, train_split, val_split, test_split):
        """Create train/val/test splits using sklearn."""
        n = len(self.files)
        indices = list(range(n))
        
        # First split: train vs temp
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=train_split, 
            random_state=42,
            stratify=self.labels
        )
        
        # Second split: val vs test from temp
        val_ratio = val_split / (val_split + test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio,
            random_state=42,
            stratify=[self.labels[i] for i in temp_idx]
        )
        
        if self.split == "train":
            self.indices = train_idx
        elif self.split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx
    
    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio file."""
        try:
            import librosa
            audio_np, sr = librosa.load(filepath, sr=None, mono=True)
            audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            
            if sr != self.sample_rate:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.sample_rate)
                audio = torch.from_numpy(audio_np).unsqueeze(0).float()
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            audio = torch.randn(1, self.target_samples)
            return audio
        
        # Pad or trim
        if audio.shape[1] < self.target_samples:
            audio = F.pad(audio, (0, self.target_samples - audio.shape[1]))
        elif audio.shape[1] > self.target_samples:
            if self.split == "train":
                start = random.randint(0, audio.shape[1] - self.target_samples)
            else:
                start = (audio.shape[1] - self.target_samples) // 2
            audio = audio[:, start:start + self.target_samples]
        
        return audio
    
    def __getitem__(self, idx: int) -> Dict:
        actual_idx = self.indices[idx]
        filepath = self.files[actual_idx]
        label = self.labels[actual_idx]
        domain = self.domains[actual_idx]
        
        try:
            audio = self._load_audio(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return self.__getitem__((idx + 1) % len(self.indices))
        
        return {
            "audio": audio.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "domain": domain,
            "filepath": filepath,
        }
    
    def __len__(self) -> int:
        return len(self.indices)


def collate_fn(batch, mel_augmenter=None):
    """Collate function for batching."""
    audios = [item["audio"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    domains = [item["domain"] for item in batch]
    filepaths = [item["filepath"] for item in batch]
    
    return {
        "audio": audios,
        "label": labels,
        "domain": domains[0] if len(set(domains)) == 1 else domains,
        "filepath": filepaths,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    data_root = "data/raw"
    
    if not Path(data_root).exists():
        print(f"Data directory not found: {data_root}")
        sys.exit(0)
    
    for split in ["train", "val", "test"]:
        dataset = CrossSpeciesDataset(
            data_root=data_root,
            split=split,
            target_duration=10.0,
            return_raw_audio=True,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n{split.upper()} split: {len(dataset)} samples")
            print(f"  Audio shape: {sample['audio'].shape}")
            print(f"  Label: {sample['label']}")
            print(f"  Domain: {sample['domain']}")