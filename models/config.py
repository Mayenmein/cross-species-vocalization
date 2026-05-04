from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
import json


# ==================================================
# MODEL CONFIG
# ==================================================
@dataclass
class ModelConfig:
    whisper_size: str = "tiny"
    encoder_dim: Optional[int] = None

    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.2

    attention_heads: int = 8
    attention_dropout: float = 0.1

    num_domains: int = 3
    num_classes: int = 20

    classifier_hidden: List[int] = field(default_factory=lambda: [256, 128])
    classifier_dropout: float = 0.3

    sample_rate: int = 16000
    n_mels: int = 80
    target_duration: float = 10.0

    use_domain_adapters: bool = True
    use_domain_confusion: bool = True
    domain_confusion_weight: float = 0.1

    def __post_init__(self):
        if self.encoder_dim is None:
            self.encoder_dim = {
                "tiny": 384,
                "small": 768,
                "medium": 1024,
            }.get(self.whisper_size.split(".")[0], 384)

    @property
    def lstm_output_dim(self):
        return self.lstm_hidden * (2 if self.lstm_bidirectional else 1)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str):
        return cls(**json.load(open(path)))


# ==================================================
# TRAINING CONFIG
# ==================================================
@dataclass
class TrainingConfig:
    # Paths
    data_root: str = "data/raw"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Phase configuration
    phase1_epochs: int = 20
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 16
    
    phase2_epochs: int = 15
    phase2_lr: float = 5e-5
    phase2_batch_size: int = 16
    phase2_layers_to_unfreeze: int = 4
    
    phase3_epochs: int = 10
    phase3_lr: float = 1e-3
    phase3_batch_size: int = 8

    # Optimization
    weight_decay: float = 0.01
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Logging
    use_wandb: bool = False

    # Hardware
    device: str = "cuda"
    num_workers: int = 2

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str):
        return cls(**json.load(open(path)))


# ==================================================
# LABELS + DOMAINS - CONSISTENT NAMING
# ==================================================
SOUND_CLASSES = [
    # Animal (0-9)
    "dog_bark_playful", "dog_growl_aggressive", "dog_whine",
    "cat_meow", "cat_hiss",
    "bird_song", "bird_alarm",
    "whale_song", "whale_click",
    "frog_chorus",
    # Human non-verbal (10-14)
    "human_cough_dry", "human_cough_wet",
    "human_cry", "human_laugh", "human_gasp",
    # Machinery (15-19)
    "engine_normal", "engine_knock",
    "bearing_fault", "alarm_siren", "hydraulic_leak"
]

DOMAINS = ["animal", "human_nonverbal", "machinery"]

DOMAIN_MAP = {
    "animal": list(range(10)),
    "human_nonverbal": list(range(10, 15)),
    "machinery": list(range(15, 20)),
}

CLASS_TO_DOMAIN = {
    c: d for d, cls in DOMAIN_MAP.items() for c in cls
}


# ==================================================
# FACTORY
# ==================================================
def get_configs(whisper_size="tiny"):
    return ModelConfig(whisper_size), TrainingConfig()


if __name__ == "__main__":
    m, t = get_configs()

    print("Model:", m)
    print("LSTM dim:", m.lstm_output_dim)
    print("Classes:", len(SOUND_CLASSES))
    print("Domains:", DOMAINS)
    print("Class to Domain:", CLASS_TO_DOMAIN)