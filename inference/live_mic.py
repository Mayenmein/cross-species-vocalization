"""
Real-time microphone classification.
"""

import sounddevice as sd
from pathlib import Path
import sys
import tempfile
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.classifier import AudioClassifier
from inference.utils import format_prediction


class LiveMicrophoneClassifier:
    """
    Real-time sound classification from microphone input.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/best.ckpt",
        duration: float = 5.0,
        sample_rate: int = 16000,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.classifier = AudioClassifier(model_path=model_path)
    
    def record_and_classify(self, domain: str = "animal") -> dict:
        """Record and classify audio from microphone."""
        print(f"\n🎤 Recording {self.duration} seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"   {i}...")
            import time
            time.sleep(0.5)
        
        print("   ▶ Recording...")
        
        # Record
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        
        print("   ✓ Recording complete!")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            result = self.classifier.classify(f.name, domain=domain)
            Path(f.name).unlink()
        
        return result
    
    def run_interactive(self, domain: str = "animal"):
        """Run interactive classification loop."""
        print("\n" + "=" * 60)
        print("LIVE SOUND CLASSIFICATION")
        print("=" * 60)
        print("Press Enter to record, 'q' to quit")
        print(f"Current domain: {domain}")
        print("=" * 60)
        
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "q":
                print("Goodbye!")
                break
            
            if cmd.startswith("domain "):
                domain = cmd[7:]
                print(f"Domain switched to: {domain}")
                continue
            
            try:
                result = self.record_and_classify(domain=domain)
                print(format_prediction(result))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live microphone classification")
    parser.add_argument("--model", type=str, default="checkpoints/best.ckpt")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--domain", type=str, default="animal")
    parser.add_argument("--once", action="store_true", help="Record once and exit")
    
    args = parser.parse_args()
    
    classifier = LiveMicrophoneClassifier(
        model_path=args.model,
        duration=args.duration,
    )
    
    if args.once:
        result = classifier.record_and_classify(domain=args.domain)
        print(format_prediction(result))
    else:
        classifier.run_interactive(domain=args.domain)