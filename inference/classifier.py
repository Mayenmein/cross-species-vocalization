"""
Single-file audio classification.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.utils import load_model, load_audio, get_top_predictions, format_prediction
from models.config import ModelConfig


class AudioClassifier:
    """
    Classify audio files using the fine-tuned model.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/best.ckpt",
        whisper_size: str = "tiny",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model, self.device = load_model(model_path, whisper_size, device)
        self.config = ModelConfig(whisper_size=whisper_size)
    
    def classify(
        self,
        filepath: str,
        domain: str = "animal",
    ) -> Dict:
        """
        Classify a single audio file.
        
        Args:
            filepath: Path to audio file
            domain: Domain hint ("animal", "human_health", "machinery", "environmental")
            
        Returns:
            Dictionary with prediction results
        """
        # Load audio
        audio = load_audio(filepath, self.config.target_duration, self.config.sample_rate)
        audio = audio.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Classify
        with torch.no_grad():
            output = self.model(audio, domain=domain)
            logits = output["logits"][0]  # Remove batch dimension
        
        # Get predictions
        top5 = get_top_predictions(logits, k=5)
        
        return {
            "filepath": filepath,
            "prediction": top5[0],
            "top5": top5,
        }
    
    def classify_folder(
        self,
        folder_path: str,
        domain: str = "animal",
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg"),
    ) -> List[Dict]:
        """
        Classify all audio files in a folder.
        """
        from tqdm import tqdm
        
        folder = Path(folder_path)
        audio_files = []
        for ext in extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"*{ext.upper()}"))
        
        audio_files = sorted(set(audio_files))
        
        results = []
        for filepath in tqdm(audio_files, desc="Classifying"):
            try:
                result = self.classify(str(filepath), domain=domain)
                results.append(result)
            except Exception as e:
                print(f"Error with {filepath.name}: {e}")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify audio files")
    parser.add_argument("input", type=str, help="Audio file or folder path")
    parser.add_argument("--model", type=str, default="checkpoints/best.ckpt")
    parser.add_argument("--domain", type=str, default="animal")
    
    args = parser.parse_args()
    
    classifier = AudioClassifier(model_path=args.model)
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = classifier.classify(str(input_path), domain=args.domain)
        print(format_prediction(result))
    else:
        results = classifier.classify_folder(str(input_path), domain=args.domain)
        for result in results:
            print(format_prediction(result))