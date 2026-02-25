"""
Real-Time Inference Script

Standalone script to run real-time rehabilitation exercise grading.
Uses webcam, MediaPipe pose extraction, and trained/simplified model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/rehab_model.keras")
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_app(model_path=args.model, use_simplified=args.simplified, config_path=args.config)
