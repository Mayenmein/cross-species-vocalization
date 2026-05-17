from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import json


# ==================================================
# 30 CLASSES ACROSS 4 DOMAINS
# ==================================================
SOUND_CLASSES = [
    # ── Animal Vocalizations (0-9) ──
    "dog_bark", "dog_growl", "dog_whine",
    "cat_meow", "cat_hiss",
    "bird_song", "bird_alarm",
    "whale_song", "whale_call",
    "frog_chorus",
    
    # ── Human Non-Verbal / Healthcare (10-17) ──
    "cough_dry", "cough_wet",
    "sneeze", "breathing_labored",
    "cry_infant", "cry_adult",
    "laugh", "gasp",
    
    # ── Machinery & Industry (18-24) ──
    "engine_normal", "engine_knock",
    "bearing_fault", "hydraulic_leak",
    "alarm_fire", "alarm_siren",
    "gearbox_grind",
    
    # ── Environmental & Context (25-29) ──
    "rain_heavy", "wind_strong",
    "thunder", "water_running",
    "fire_crackling",
]

DOMAINS = [
    "animal",
    "human_health",
    "machinery",
    "environmental",
]

DOMAIN_MAP = {
    "animal": list(range(0, 10)),
    "human_health": list(range(10, 18)),
    "machinery": list(range(18, 25)),
    "environmental": list(range(25, 30)),
}

CLASS_TO_DOMAIN = {
    c: d for d, cls in DOMAIN_MAP.items() for c in cls
}


@dataclass
class ModelConfig:
    whisper_size: str = "tiny"
    encoder_dim: Optional[int] = None

    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.4

    attention_heads: int = 8
    attention_dropout: float = 0.1

    num_domains: int = 4
    num_classes: int = 30

    classifier_hidden: List[int] = field(default_factory=lambda: [64, 32])
    classifier_dropout: float = 0.5

    sample_rate: int = 16000
    n_mels: int = 80
    target_duration: float = 10.0

    use_domain_adapters: bool = True
    use_domain_confusion: bool = True
    domain_confusion_weight: float = 0.1

    # LoRA config (using PEFT)
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

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


@dataclass
class TrainingConfig:
    data_root: str = "data/raw"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Training
    max_epochs: int = 45  # Total epochs (phase1+phase2 combined)
    learning_rate: float = 1e-4
    batch_size: int = 16
    weight_decay: float = 0.1
    
    # Multi-stage learning
    phase1_epochs: int = 20
    phase2_epochs: int = 15
    phase3_epochs: int = 10
    
    early_stopping_patience: int = 3 
    use_wandb = True

    phase1_lr: float = 5e-4
    phase2_lr: float = 2e-4
    phase3_learning_rate: float = 1e-4

    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Hardware
    device: str = "cuda"
    num_workers: int = 2

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str):
        return cls(**json.load(open(path)))


def get_configs(whisper_size="tiny"):
    return ModelConfig(whisper_size), TrainingConfig()


if __name__ == "__main__":
    m, t = get_configs()
    print("Model:", m)
    print("LSTM dim:", m.lstm_output_dim)
    print("Classes:", len(SOUND_CLASSES))
    print("Domains:", DOMAINS)