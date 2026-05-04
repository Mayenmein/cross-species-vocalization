#!/usr/bin/env python3
"""
run_training.py

Small helper to run the three fine-tuning phases defined in
`training/fine_tune.FineTuner`.

Usage examples:
  python run_training.py --phase 1
  python run_training.py --phase all --whisper tiny
"""

import argparse

from models.config import get_configs
from models.cross_species_model import CrossSpeciesVocalizationModel
from training.fine_tune import FineTuner


def main():
    parser = argparse.ArgumentParser(description="Run training phases for cross-species model")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all",
                        help="Which phase to run (1, 2, 3, or all)")
    parser.add_argument("--whisper", default="tiny", help="Whisper model size: tiny | small | medium")
    parser.add_argument("--device", default=None, help="Force device (e.g. cpu or cuda)")

    args = parser.parse_args()

    mcfg, tcfg = get_configs(whisper_size=args.whisper)
    if args.device:
        tcfg.device = args.device

    model = CrossSpeciesVocalizationModel(mcfg)
    tuner = FineTuner(model, mcfg, tcfg)

    # Run requested phases
    if args.phase in ("1", "all"):
        print("Starting Phase 1")
        tuner.phase1()

    if args.phase in ("2", "all"):
        print("Starting Phase 2")
        tuner.phase2()

    if args.phase in ("3", "all"):
        print("Starting Phase 3")
        tuner.phase3()

    print("Done")


if __name__ == "__main__":
    main()
