#!/usr/bin/env python3
"""
Compatibility wrapper. The predictor now lives in the package:

    from gender_predict import GenderPredictor

Command line (same as ``gender-predict`` after ``pip install -e .``):

    python scripts/final_predictor.py "Maria Rossi"
    python scripts/final_predictor.py --input data.csv --output results.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gender_predict.inference.cli import main  # noqa: E402
from gender_predict.inference import GenderPredictor  # noqa: E402,F401

FinalGenderPredictor = GenderPredictor  # backward-compatible alias

if __name__ == "__main__":
    sys.exit(main())
