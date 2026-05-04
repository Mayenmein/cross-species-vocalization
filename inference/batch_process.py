# inference/batch_process.py
"""
Batch process entire folders of audio files.

Process hundreds of files at once with progress tracking,
summary statistics, and CSV export.

Usage:
    python inference/batch_process.py input_folder/ --output results.csv
    python inference/batch_process.py input_folder/ --domain animal --workers 4
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import time
import sys
import json
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.classifier import AudioClassifier
from models.config import SOUND_CLASSES, DOMAIN_MAP


class BatchProcessor:
    """
    Process large numbers of audio files efficiently.
    
    Features:
    - Multi-threaded processing for speed
    - Progress bar with ETA
    - Summary statistics
    - CSV/JSON export
    - Error handling for corrupt files
    """
    
    def __init__(
        self,
        model_path: str = "checkpoints/final_model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.classifier = AudioClassifier(
            model_path=model_path,
            device=device,
        )
        
        self.results = []
        self.errors = []
        self.start_time = None
    
    def process_folder(
        self,
        input_folder: str,
        output_file: Optional[str] = None,
        domain: str = "auto",
        num_workers: int = 1,
        recursive: bool = True,
        extensions: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
    ) -> List[Dict]:
        """
        Process all audio files in a folder.
        
        Args:
            input_folder: Path to folder with audio files
            output_file: Path to save results (CSV or JSON)
            domain: Domain hint or "auto"
            num_workers: Number of parallel workers
            recursive: Search subfolders
            extensions: Audio file extensions to process
            
        Returns:
            List of classification results
        """
        self.start_time = time.time()
        
        # Find all audio files
        audio_files = self._find_audio_files(
            input_folder, extensions, recursive
        )
        
        if not audio_files:
            print(f"No audio files found in {input_folder}")
            return []
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"Files found:    {len(audio_files)}")
        print(f"Output:         {output_file or 'None'}")
        print(f"Domain:         {domain}")
        print(f"Workers:        {num_workers}")
        print(f"{'='*60}\n")
        
        # Process files
        if num_workers > 1:
            self.results = self._process_parallel(
                audio_files, domain, num_workers
            )
        else:
            self.results = self._process_sequential(audio_files, domain)
        
        # Compute summary
        summary = self._compute_summary()
        self._print_summary(summary)
        
        # Save results
        if output_file:
            self._save_results(output_file)
        
        return self.results
    
    def _find_audio_files(
        self,
        folder: str,
        extensions: tuple,
        recursive: bool,
    ) -> List[Path]:
        """Find all audio files in folder."""
        folder = Path(folder)
        audio_files = []
        
        if recursive:
            for ext in extensions:
                audio_files.extend(folder.rglob(f"*{ext}"))
                audio_files.extend(folder.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                audio_files.extend(folder.glob(f"*{ext}"))
                audio_files.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted(set(audio_files))
    
    def _process_sequential(
        self,
        files: List[Path],
        domain: str,
    ) -> List[Dict]:
        """Process files one by one with progress bar."""
        results = []
        total = len(files)
        
        for i, filepath in enumerate(files):
            # Progress
            elapsed = time.time() - self.start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            
            print(f"\r[{i+1:4d}/{total}] {filepath.name[:50]:50s} "
                  f"| {rate:.1f} files/s | ETA: {eta:.0f}s", end="")
            
            try:
                result = self.classifier.classify(str(filepath), domain=domain)
                results.append(result)
            except Exception as e:
                self.errors.append({
                    "file": str(filepath),
                    "error": str(e),
                })
        
        print()  # New line after progress
        return results
    
    def _process_parallel(
        self,
        files: List[Path],
        domain: str,
        num_workers: int,
    ) -> List[Dict]:
        """Process files in parallel using thread pool."""
        results = []
        total = len(files)
        
        def process_one(filepath):
            try:
                return self.classifier.classify(str(filepath), domain=domain)
            except Exception as e:
                self.errors.append({
                    "file": str(filepath),
                    "error": str(e),
                })
                return None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_one, fp): fp
                for fp in files
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    results.append(result)
                
                print(f"\r[{completed:4d}/{total}] Processing...", end="")
        
        print()
        return results
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics from results."""
        if not self.results:
            return {}
        
        # Class distribution
        class_counts = Counter()
        domain_counts = Counter()
        confidences = []
        
        for r in self.results:
            pred = r["prediction"]
            class_counts[pred["class_name"]] += 1
            domain_counts[pred["domain"]] += 1
            confidences.append(pred["confidence"])
        
        # Most common predictions
        most_common = class_counts.most_common(10)
        
        return {
            "total_files": len(self.results) + len(self.errors),
            "successful": len(self.results),
            "errors": len(self.errors),
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0,
            "class_distribution": dict(class_counts),
            "domain_distribution": dict(domain_counts),
            "most_common_predictions": [
                {"class": cls, "count": count}
                for cls, count in most_common
            ],
            "processing_time_s": time.time() - self.start_time,
        }
    
    def _print_summary(self, summary: Dict):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files:     {summary.get('total_files', 0)}")
        print(f"Successful:      {summary.get('successful', 0)}")
        print(f"Errors:          {summary.get('errors', 0)}")
        print(f"Time:            {summary.get('processing_time_s', 0):.1f}s")
        print(f"Avg confidence:  {summary.get('avg_confidence', 0):.2%}")
        
        print(f"\nTop Predictions:")
        for item in summary.get("most_common_predictions", [])[:5]:
            bar = "█" * min(40, item["count"])
            print(f"  {item['class']:<30s} {bar} ({item['count']})")
        
        print(f"\nDomain Distribution:")
        for domain, count in summary.get("domain_distribution", {}).items():
            print(f"  {domain:<20s}: {count}")
    
    def _save_results(self, output_file: str):
        """Save results to CSV or JSON."""
        output_path = Path(output_file)
        
        if output_path.suffix == ".csv":
            self._save_csv(output_path)
        elif output_path.suffix == ".json":
            self._save_json(output_path)
        else:
            # Default to CSV
            self._save_csv(output_path.with_suffix(".csv"))
    
    def _save_csv(self, path: Path):
        """Save results as CSV."""
        rows = []
        for r in self.results:
            pred = r["prediction"]
            rows.append({
                "filepath": r["filepath"],
                "predicted_class": pred["class_name"],
                "confidence": pred["confidence"],
                "domain": pred["domain"],
                "top3_classes": "|".join(
                    f"{t['class_name']}:{t['confidence']:.2f}"
                    for t in r["top5"][:3]
                ),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"\nResults saved to {path} ({len(rows)} rows)")
    
    def _save_json(self, path: Path):
        """Save results as JSON."""
        # Simplify results for JSON
        simplified = []
        for r in self.results:
            simplified.append({
                "filepath": r["filepath"],
                "prediction": r["prediction"],
                "top3": r["top5"][:3],
            })
        
        with open(path, "w") as f:
            json.dump({
                "results": simplified,
                "summary": self._compute_summary(),
                "errors": self.errors,
            }, f, indent=2)
        
        print(f"\nResults saved to {path}")


# ─── CLI ───
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process audio files")
    parser.add_argument("input_folder", type=str, help="Folder with audio files")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file (CSV or JSON)")
    parser.add_argument("--model", type=str, default="checkpoints/final_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--domain", type=str, default="auto",
                       choices=["animal", "human_nonverbal", "machinery", "auto"],
                       help="Domain hint")
    parser.add_argument("--workers", "-w", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Don't search subfolders")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(model_path=args.model)
    
    results = processor.process_folder(
        input_folder=args.input_folder,
        output_file=args.output,
        domain=args.domain,
        num_workers=args.workers,
        recursive=not args.no_recursive,
    )