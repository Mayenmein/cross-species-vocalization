"""
Batch process folders of audio files.
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter 
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.classifier import AudioClassifier


class BatchProcessor:
    """
    Efficient batch processing of audio files.
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/best.ckpt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.classifier = AudioClassifier(model_path=model_path, device=device)
        self.results = []
        self.errors = []
    
    def process_folder(
        self,
        input_folder: str,
        output_file: Optional[str] = None,
        domain: str = "animal",
        num_workers: int = 1,
        recursive: bool = True,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg"),
    ) -> List[Dict]:
        """Process all audio files in a folder."""
        
        # Find files
        audio_files = self._find_audio_files(input_folder, extensions, recursive)
        
        if not audio_files:
            print(f"No audio files found in {input_folder}")
            return []
        
        print(f"\nFound {len(audio_files)} files")
        print(f"Domain: {domain}, Workers: {num_workers}")
        
        # Process
        if num_workers > 1:
            self.results = self._process_parallel(audio_files, domain, num_workers)
        else:
            self.results = self._process_sequential(audio_files, domain)
        
        # Summary
        self._print_summary()
        
        # Save
        if output_file:
            self._save_results(output_file)
        
        return self.results
    
    def _find_audio_files(self, folder: str, extensions: tuple, recursive: bool) -> List[Path]:
        """Find all audio files."""
        folder = Path(folder)
        files = []
        
        for ext in extensions:
            if recursive:
                files.extend(folder.rglob(f"*{ext}"))
                files.extend(folder.rglob(f"*{ext.upper()}"))
            else:
                files.extend(folder.glob(f"*{ext}"))
                files.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted(set(files))
    
    def _process_sequential(self, files: List[Path], domain: str) -> List[Dict]:
        """Process files sequentially with progress bar."""
        results = []
        
        for filepath in tqdm(files, desc="Processing"):
            try:
                result = self.classifier.classify(str(filepath), domain=domain)
                results.append(result)
            except Exception as e:
                self.errors.append({"file": str(filepath), "error": str(e)})
        
        return results
    
    def _process_parallel(self, files: List[Path], domain: str, num_workers: int) -> List[Dict]:
        """Process files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.classifier.classify, str(fp), domain): fp
                for fp in files
            }
            
            for future in tqdm(as_completed(futures), total=len(files), desc="Processing"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    fp = futures[future]
                    self.errors.append({"file": str(fp), "error": str(e)})
        
        return results
    
    def _print_summary(self):
        """Print summary statistics."""
        if not self.results:
            return
        
        class_counts = Counter()
        confidences = []
        
        for r in self.results:
            pred = r["prediction"]
            class_counts[pred["class_name"]] += 1
            confidences.append(pred["confidence"])
        
        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Successful: {len(self.results)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Avg confidence: {sum(confidences)/len(confidences):.2%}")
        
        print(f"\nTop 5 classes:")
        for cls, count in class_counts.most_common(5):
            print(f"  {cls}: {count}")
    
    def _save_results(self, output_file: str):
        """Save results to CSV or JSON."""
        rows = []
        for r in self.results:
            pred = r["prediction"]
            rows.append({
                "filepath": r["filepath"],
                "predicted_class": pred["class_name"],
                "confidence": pred["confidence"],
                "domain": pred["domain"],
            })
        
        df = pd.DataFrame(rows)
        
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        elif output_file.endswith(".json"):
            df.to_json(output_file, orient="records", indent=2)
        else:
            df.to_csv(output_file + ".csv", index=False)
        
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process audio files")
    parser.add_argument("input_folder", type=str)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--model", type=str, default="checkpoints/best.ckpt")
    parser.add_argument("--domain", type=str, default="animal")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--no-recursive", action="store_true")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(model_path=args.model)
    
    processor.process_folder(
        input_folder=args.input_folder,
        output_file=args.output,
        domain=args.domain,
        num_workers=args.workers,
        recursive=not args.no_recursive,
    )