# inference/live_mic.py
"""
Real-time microphone classification.

Captures audio from microphone, classifies it live, and displays results.
Press Enter to start recording, Ctrl+C to exit.

Usage:
    python inference/live_mic.py
    python inference/live_mic.py --duration 5 --domain animal
"""

import torch
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Dict
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.classifier import AudioClassifier, format_result


class LiveMicrophoneClassifier:
    """
    Real-time sound classification from microphone input.
    
    Captures audio in chunks, converts to mel spectrograms,
    and classifies using the trained model.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/final_model.pt",
        duration: float = 5.0,
        sample_rate: int = 16000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_path: Path to trained model
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
            device: Device for inference
        """
        self.duration = duration
        self.sample_rate = sample_rate
        
        # Load classifier
        self.classifier = AudioClassifier(
            model_path=model_path,
            device=device,
        )
        
        # List available devices
        self._list_devices()
    
    def _list_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']})")
        print("-" * 50)
    
    def record_and_classify(
        self,
        domain: str = "auto",
        show_waveform: bool = False,
    ) -> Dict:
        """
        Record audio from microphone and classify it.
        
        Args:
            domain: Domain hint for classification
            show_waveform: Print a simple text waveform
            
        Returns:
            Classification result
        """
        print(f"\n🎤 Recording {self.duration} seconds...")
        print("   Make a sound! (animal call, cough, machine noise, etc.)")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(0.5)
        
        print("   ▶ Recording...")
        
        # Record audio
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        
        print("   ✓ Recording complete!")
        
        # Show waveform
        if show_waveform:
            self._print_waveform(audio)
        
        # Convert to tensor and classify
        # The classifier expects a file path, so we save to temp
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            result = self.classifier.classify(f.name, domain=domain)
            Path(f.name).unlink()  # Clean up
        
        return result
    
    def run_interactive(self, domain: str = "auto"):
        """
        Run interactive classification loop.
        
        Press Enter to classify, 'q' to quit.
        """
        print("\n" + "=" * 60)
        print("LIVE SOUND CLASSIFICATION")
        print("=" * 60)
        print("Press Enter to start recording")
        print("Type 'q' and Enter to quit")
        print("Type 'd animal' to switch to animal domain")
        print("=" * 60)
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == "q":
                print("Goodbye!")
                break
            
            if command.startswith("d "):
                domain = command[2:].strip()
                print(f"Domain switched to: {domain}")
                continue
            
            # Classify
            try:
                result = self.record_and_classify(domain=domain)
                print(format_result(result))
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_waveform(self, audio: np.ndarray):
        """Print a simple text waveform visualization."""
        audio_flat = audio.flatten()
        
        # Downsample for display
        display_points = 80
        step = max(1, len(audio_flat) // display_points)
        downsampled = audio_flat[::step][:display_points]
        
        # Normalize
        max_val = np.abs(downsampled).max()
        if max_val > 0:
            downsampled = downsampled / max_val
        
        print("\n   Waveform:")
        for val in downsampled:
            bar_len = int(abs(val) * 20)
            bar = "█" * bar_len
            print(f"   {'▁' if val < 0 else '▔'} {bar}")


# ─── CLI ───
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live microphone classification")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Recording duration in seconds")
    parser.add_argument("--domain", type=str, default="auto",
                       choices=["animal", "human_nonverbal", "machinery", "auto"],
                       help="Domain hint")
    parser.add_argument("--once", action="store_true",
                       help="Record once and exit (no interactive mode)")
    
    args = parser.parse_args()
    
    classifier = LiveMicrophoneClassifier(
        model_path=args.model,
        duration=args.duration,
    )
    
    if args.once:
        result = classifier.record_and_classify(domain=args.domain)
        print(format_result(result))
    else:
        classifier.run_interactive(domain=args.domain)