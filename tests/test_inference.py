"""
Regression tests for the inference API.

``tests/regression/baseline_v4r1.json`` holds the probabilities produced by
the pipeline selected in June 2025 (V4-R1 weights + transliteration) on a
sample of 300 names from the comparison set plus a few non-Latin and edge
cases. Any change to preprocessing, transliteration or model loading must
keep these probabilities unchanged (tolerance 1e-6).

Run with:  pytest tests/  (or  python tests/test_inference.py)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gender_predict import GenderPredictor  # noqa: E402

BASELINE = ROOT / "tests" / "regression" / "baseline_v4r1.json"
TOL = 1e-6


def _predictor():
    return GenderPredictor(ROOT / "models" / "production", device="cpu")


def test_baseline_probabilities():
    rows = json.loads(BASELINE.read_text(encoding="utf-8"))
    p = _predictor()
    bad = []
    for row in rows:
        r = p.predict(row["name"])
        if abs(r["probability_female"] - row["prob_female"]) > TOL or r["predicted_gender"] != row["gender"]:
            bad.append((row["name"], row["prob_female"], r["probability_female"]))
    assert not bad, f"{len(bad)} names changed, e.g. {bad[:5]}"


def test_dataframe_api():
    import pandas as pd

    df = pd.DataFrame({"primaryName": ["Maria Rossi", "Mario Rossi", None]})
    out = _predictor().predict_dataframe(df)
    assert list(out["predicted_gender"][:2]) == ["W", "M"]
    assert {"predicted_gender", "probability_female", "confidence"} <= set(out.columns)


def test_cli_single(capsys):
    from gender_predict.inference.cli import main

    assert main(["--model-dir", str(ROOT / "models" / "production"), "-q", "Maria Rossi"]) == 0
    assert "Maria Rossi\tW" in capsys.readouterr().out


if __name__ == "__main__":
    test_baseline_probabilities()
    test_dataframe_api()
    print("ok")
