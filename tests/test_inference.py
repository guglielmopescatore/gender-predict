"""
Regression tests for the inference API.

``tests/regression/baseline_4_0.json`` holds the answers of the production model (models/production, version 4.0)
on a sample of 300 names from the 2025 comparison set plus non-Latin and edge cases: probability, gender, status
and reliability. ``tests/regression/baseline_v4r1.json`` holds the probabilities of the previous production model
(models/v4r1, June 2025). Any change to normalisation, tables or model loading must keep both unchanged
(tolerance 1e-6), unless the change is intended and the baseline is regenerated.

Run with:  pytest tests/  (or  python tests/test_inference.py)
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gender_predict import GenderPredictor  # noqa: E402

BASELINE = ROOT / "tests" / "regression" / "baseline_4_0.json"
BASELINE_V4R1 = ROOT / "tests" / "regression" / "baseline_v4r1.json"
TOL = 1e-6


def _predictor():
    return GenderPredictor(ROOT / "models" / "production", device="cpu")


def _legacy():
    return GenderPredictor(ROOT / "models" / "v4r1", device="cpu")


def test_baseline_production():
    rows = json.loads(BASELINE.read_text(encoding="utf-8"))
    p = _predictor()
    bad = []
    for row, r in zip(rows, p.predict_many([row["name"] for row in rows])):
        prob = r["probability_female"]
        same = (row["prob_female"] is None and prob != prob) or (row["prob_female"] is not None and abs(prob - row["prob_female"]) <= TOL)
        if not same or (r["predicted_gender"], r["status"], r.get("reliability")) != (row["gender"], row["status"], row["reliability"]):
            bad.append((row["name"], row["prob_female"], prob, r["status"]))
    assert not bad, f"{len(bad)} names changed, e.g. {bad[:5]}"


def test_baseline_v4r1():
    """The previous production model must keep giving the probabilities of June 2025."""
    rows = json.loads(BASELINE_V4R1.read_text(encoding="utf-8"))
    p = _legacy()
    bad = []
    for row in rows:
        r = p.predict(row["name"])
        if abs(r["probability_female"] - row["prob_female"]) > TOL or r["predicted_gender"] != row["gender"]:
            bad.append((row["name"], row["prob_female"], r["probability_female"]))
    assert not bad, f"{len(bad)} names changed, e.g. {bad[:5]}"


def test_attention_heads_match_training():
    """A model must be rebuilt with the number of attention heads it was trained with (8 for both models).
    The weights load with any value, so a mismatch is silent: it only changes the probabilities."""
    old = _legacy()
    old.load()
    params = json.loads((ROOT / "models" / "v4r1" / "parameters.json").read_text(encoding="utf-8"))
    assert old.model.firstname_attention.num_heads == params["num_heads"] == 8
    assert old.model.lastname_attention.num_heads == params["num_heads"]
    new = _predictor()
    new.load()
    params = json.loads((ROOT / "models" / "production" / "parameters.json").read_text(encoding="utf-8"))
    assert new.model.name_attention.num_heads == params["num_heads"] == 8


def test_undetermined_names():
    """Names that cannot be represented are not predicted, and say why."""
    p = _predictor()
    for name, status in (("", "empty"), (None, "empty"), ("J. Smith", "initial_only"), ("محمد علي", "unsupported_script")):
        r = p.predict(name)
        assert r["predicted_gender"] == "Unknown" and r["status"] == status and r["probability_female"] != r["probability_female"]
    for name, gender in (("Γιώργος Παπαδόπουλος", "M"), ("佐藤花子", "W"), ("김민석", "M"), ("王芳", "W"), ("Анна Каренина", "W")):
        assert p.predict(name)["predicted_gender"] == gender, name


def test_batch_matches_single():
    """predict_many (batched forward pass) must give the same probabilities as predict, name by name."""
    p = _predictor()
    names = ["Maria Rossi", "Andrea Bianchi", "John Smith", "Madonna", "", "Jean-Francois D'Arc", "Иван Петров", "A", None,
             "Anna Maria De Luca", "Kim Min-Seok", "Nguyen Van A"] * 3
    single = [p.predict(n)["probability_female"] for n in names]
    for bs in (1, 5, 1024):
        batch = p.predict_many(names, batch_size=bs)
        assert len(batch) == len(names)
        assert all((a != a and b["probability_female"] != b["probability_female"]) or abs(a - b["probability_female"]) < 1e-5
                   for a, b in zip(single, batch))


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
    test_baseline_production()
    test_baseline_v4r1()
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
