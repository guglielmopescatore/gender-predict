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


def test_attention_heads_match_training():
    """The model must be rebuilt with the number of attention heads it was trained with (8 for V4-R1).
    The weights load with any value, so a mismatch is silent: it only changes the probabilities."""
    p = _predictor()
    p.load()
    params = json.loads((ROOT / "models" / "production" / "parameters.json").read_text(encoding="utf-8"))
    assert p.model.firstname_attention.num_heads == params["num_heads"] == 8
    assert p.model.lastname_attention.num_heads == params["num_heads"]


def test_batch_matches_single():
    """predict_many (batched forward pass) must give the same probabilities as predict, name by name."""
    p = _predictor()
    names = ["Maria Rossi", "Andrea Bianchi", "John Smith", "Madonna", "", "Jean-Francois D'Arc", "Иван Петров", "A", None,
             "Anna Maria De Luca", "Kim Min-Seok", "Nguyen Van A"] * 3
    single = [p.predict(n)["probability_female"] for n in names]
    for bs in (1, 5, 1024):
        batch = p.predict_many(names, batch_size=bs)
        assert len(batch) == len(names)
        assert max(abs(a - b["probability_female"]) for a, b in zip(single, batch)) < 1e-5


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


def test_declared_country():
    """The country is optional: without it nothing changes; with it a clear-cut name follows the country's table."""
    p = _predictor()
    plain = p.predict("Andrea Rossi")
    assert "source" not in plain
    it = p.predict("Andrea Rossi", country="IT")
    assert it["source"] == "country_table" and it["predicted_gender"] == "M" and it["model_probability_female"] == plain["probability_female"]
    many = p.predict_many(["Andrea Rossi", "Simone Martin", "Zzzqqq Rossi"], country=["it", "FR", "IT"])
    assert [r["source"] for r in many] == ["country_table", "country_table", "model"] and many[1]["predicted_gender"] == "W"
    assert p.predict_many(["Andrea Rossi"], country=[None])[0]["probability_female"] == plain["probability_female"]
