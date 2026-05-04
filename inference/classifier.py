# inference/classifier.py
"""
Single-file audio classification.

Quick classification of any audio file using the trained model.
Supports WAV, MP3, FLAC, OGG formats.

Usage:
    python inference/classifier.py path/to/audio.wav
    python inference/classifier.py path/to/audio.wav --domain animal
    python inference/classifier.py path/to/folder/  (classify all audio in folder)
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cross_species_model import CrossSpeciesVocalizationModel
from models.config import ModelConfig, SOUND_CLASSES, DOMAIN_MAP, CLASS_TO_DOMAIN


class AudioClassifier:
    """
    Classify a single audio file using the fine-tuned model.
    
    Handles preprocessing, inference, and result formatting.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/final_model.pt",
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_config: Model configuration (loads from checkpoint if None)
            device: Device to run inference on
        """
        self.device = device
        
        # Load model
        if model_config is None:
            model_config = ModelConfig(whisper_size="tiny")
        
        self.model = CrossSpeciesVocalizationModel(model_config)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}. Using untrained model.")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Audio preprocessing
        self.sample_rate = model_config.sample_rate
        self.target_duration = model_config.target_duration
        self.target_samples = int(self.target_duration * self.sample_rate)
        
        self.mel_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=model_config.n_mels,
        )
        
        self.db_converter = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80.0,
        )
    
    def classify(
        self,
        filepath: str,
        domain: str = "animal",
        return_attention: bool = False,
    ) -> Dict:
        """
        Classify a single audio file.
        
        Args:
            filepath: Path to audio file
            domain: Domain hint ("animal", "human_nonverbal", "machinery", or "auto")
            return_attention: If True, return attention weights for visualization
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess audio
        audio = self._load_audio(filepath)
        mel = self._audio_to_mel(audio)
        mel = mel.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            output = self.model(mel, domain=domain, return_attention=return_attention)
            
            # Get probabilities
            probs = torch.softmax(output["logits"], dim=-1)[0]
            
            # Top-5 predictions
            top5_probs, top5_indices = probs.topk(min(5, len(probs)))
            
            # Top prediction
            top_class = top5_indices[0].item()
            top_confidence = top5_probs[0].item()
            top_class_name = SOUND_CLASSES.get(top_class, f"class_{top_class}")
            top_domain = CLASS_TO_DOMAIN.get(top_class, "unknown")
        
        result = {
            "filepath": filepath,
            "prediction": {
                "class_id": top_class,
                "class_name": top_class_name,
                "confidence": top_confidence,
                "domain": top_domain,
            },
            "top5": [
                {
                    "class_id": idx.item(),
                    "class_name": SOUND_CLASSES.get(idx.item(), f"class_{idx.item()}"),
                    "confidence": prob.item(),
                    "domain": CLASS_TO_DOMAIN.get(idx.item(), "unknown"),
                }
                for idx, prob in zip(top5_indices, top5_probs)
            ],
        }
        
        if return_attention and "attention_weights" in output:
            result["attention_weights"] = output["attention_weights"][0].cpu().tolist()
        
        return result
    
    def classify_folder(
        self,
        folder_path: str,
        domain: str = "auto",
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ) -> List[Dict]:
        """
        Classify all audio files in a folder.
        
        Args:
            folder_path: Path to folder containing audio files
            domain: Domain hint or "auto" to use best guess
            extensions: Audio file extensions to process
            
        Returns:
            List of classification results
        """
        folder = Path(folder_path)
        audio_files = []
        
        for ext in extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"*{ext.upper()}"))
        
        audio_files = sorted(set(audio_files))
        
        results = []
        for i, filepath in enumerate(audio_files):
            print(f"[{i+1}/{len(audio_files)}] {filepath.name}")
            result = self.classify(str(filepath), domain=domain)
            results.append(result)
        
        return results
    
    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load audio file and preprocess."""
        try:
            audio, sr = torchaudio.load(filepath)
        except Exception as e:
            # Try with soundfile as fallback
            import soundfile as sf
            data, sr = sf.read(filepath)
            audio = torch.from_numpy(data.T).float()
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Pad or trim
        if audio.shape[1] < self.target_samples:
            padding = self.target_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif audio.shape[1] > self.target_samples:
            start = (audio.shape[1] - self.target_samples) // 2
            audio = audio[:, start:start + self.target_samples]
        
        return audio
    
    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        mel_spec = self.mel_converter(audio)
        mel_spec_db = self.db_converter(mel_spec)
        mel_spec_db = (mel_spec_db + 40) / 40  # Normalize
        
        return mel_spec_db.squeeze(0)


def format_result(result: Dict) -> str:
    """Format classification result for display."""
    pred = result["prediction"]
    
    output = f"""
╔══════════════════════════════════════════════╗
║         SOUND CLASSIFICATION RESULT          ║
╠══════════════════════════════════════════════╣
║ File:    {Path(result['filepath']).name:<35s} ║
║                                              ║
║ ┌───────────── PRIMARY PREDICTION ─────────┐ ║
║ │ Class:      {pred['class_name']:<28s} │ ║
║ │ Confidence: {pred['confidence']:.1%}                        │ ║
║ │ Domain:     {pred['domain']:<28s} │ ║
║ └──────────────────────────────────────────┘ ║
║                                              ║
║ ┌───────────── TOP 5 PREDICTIONS ──────────┐ ║"""
    
    for i, top in enumerate(result["top5"]):
        bar = "█" * int(top["confidence"] * 20) + "░" * (20 - int(top["confidence"] * 20))
        output += f"\n║ {i+1}. {top['class_name']:<28s} {bar} {top['confidence']:.1%} ║"
    
    output += """
║ └──────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════╝
"""
    return output


# ─── CLI ───
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify audio files")
    parser.add_argument("input", type=str, help="Audio file or folder path")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--domain", type=str, default="auto",
                       choices=["animal", "human_nonverbal", "machinery", "auto"],
                       help="Domain hint")
    parser.add_argument("--attention", action="store_true",
                       help="Return attention weights")
    
    args = parser.parse_args()
    
    classifier = AudioClassifier(model_path=args.model)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = classifier.classify(str(input_path), domain=args.domain, return_attention=args.attention)
        print(format_result(result))
    elif input_path.is_dir():
        results = classifier.classify_folder(str(input_path), domain=args.domain)
        for result in results:
            print(format_result(result))
    else:
        print(f"Input not found: {args.input}")