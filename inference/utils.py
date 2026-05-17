"""
Shared utilities for inference and evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig, SOUND_CLASSES, DOMAIN_MAP, CLASS_TO_DOMAIN
from training.lightning_module import AudioLightningModule


def load_model(
    checkpoint_path: str = "checkpoints/best.ckpt",
    whisper_size: str = "tiny",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load trained model from PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        whisper_size: Whisper model size (if checkpoint doesn't have config)
        device: Device to load model to
        
    Returns:
        Loaded model in eval mode
    """
    if Path(checkpoint_path).exists():
        # Load from Lightning checkpoint
        lightning_model = AudioLightningModule.load_from_checkpoint(
            checkpoint_path,
            model_config=ModelConfig(whisper_size=whisper_size),
            train_config=None,  # Not needed for inference
        )
        model = lightning_model.model
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        # Fallback to untrained model
        print(f"⚠️ Checkpoint not found at {checkpoint_path}, using untrained model")
        config = ModelConfig(whisper_size=whisper_size)
        from models.cross_species_model import CrossSpeciesVocalizationModel
        model = CrossSpeciesVocalizationModel(config)
    
    model = model.to(device)
    model.eval()
    return model, device


def load_audio(
    filepath: str,
    target_duration: float = 10.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load and preprocess audio file.
    
    Args:
        filepath: Path to audio file
        target_duration: Target duration in seconds
        sample_rate: Target sample rate
        
    Returns:
        Audio tensor of shape [samples]
    """
    target_samples = int(target_duration * sample_rate)
    
    try:
        import librosa
        audio_np, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Resample if needed
        if sr != sample_rate:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=sample_rate)
        
        audio = torch.from_numpy(audio_np).float()
        
    except Exception as e:
        # Fallback to torchaudio
        import torchaudio
        audio, sr = torchaudio.load(filepath)
        
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)
    
    # Pad or trim to target length
    if len(audio) < target_samples:
        audio = torch.nn.functional.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
    
    return audio


def get_top_predictions(
    logits: torch.Tensor,
    k: int = 5,
) -> List[Dict]:
    """
    Get top-k predictions from model logits.
    
    Args:
        logits: Model output logits [num_classes]
        k: Number of top predictions
        
    Returns:
        List of dicts with class_id, class_name, confidence, domain
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(min(k, len(probs)))
    
    predictions = []
    for idx, prob in zip(top_indices, top_probs):
        class_id = idx.item()
        predictions.append({
            "class_id": class_id,
            "class_name": SOUND_CLASSES.get(class_id, f"class_{class_id}"),
            "confidence": prob.item(),
            "domain": CLASS_TO_DOMAIN.get(class_id, "unknown"),
        })
    
    return predictions


def format_prediction(prediction: Dict, show_top5: bool = True) -> str:
    """
    Format prediction results for display.
    
    Args:
        prediction: Prediction dict from classify()
        show_top5: Whether to show top-5 predictions
        
    Returns:
        Formatted string
    """
    pred = prediction["prediction"]
    
    output = f"""
╔══════════════════════════════════════════════╗
║         SOUND CLASSIFICATION RESULT          ║
╠══════════════════════════════════════════════╣
║ File:    {Path(prediction['filepath']).name:<35s} ║
║                                              ║
║ ┌───────────── PRIMARY PREDICTION ─────────┐ ║
║ │ Class:      {pred['class_name']:<28s} │ ║
║ │ Confidence: {pred['confidence']:.1%}                        │ ║
║ │ Domain:     {pred['domain']:<28s} │ ║
║ └──────────────────────────────────────────┘ ║"""
    
    if show_top5 and "top5" in prediction:
        output += f"""
║                                              ║
║ ┌───────────── TOP 5 PREDICTIONS ──────────┐ ║"""
        
        for i, top in enumerate(prediction["top5"]):
            bar = "█" * int(top["confidence"] * 20) + "░" * (20 - int(top["confidence"] * 20))
            output += f"\n║ {i+1}. {top['class_name']:<28s} {bar} {top['confidence']:.1%} ║"
    
    output += """
║ └──────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════╝
"""
    return output


def compute_metrics(predictions: List[int], targets: List[int], num_classes: int) -> Dict:
    """
    Compute classification metrics using sklearn.
    
    Args:
        predictions: List of predicted class IDs
        targets: List of true class IDs
        num_classes: Total number of classes
        
    Returns:
        Dict with accuracy, per-class metrics, confusion matrix
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    preds = np.array(predictions)
    labels = np.array(targets)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(num_classes), zero_division=0
    )
    
    per_class = {}
    for i in range(num_classes):
        if support[i] > 0:
            class_name = SOUND_CLASSES.get(i, f"class_{i}")
            per_class[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
    
    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(num_classes)).tolist(),
    }