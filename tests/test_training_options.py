"""
Tests for options of the training data pipeline that do not need a training run.

``order_swap_prob`` (scripts/train_model.py --order_swap_prob): in training, a name of exactly two words is shown
family name first with the given probability, with the same target; validation and test data are never changed.

Run with:  pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gender_predict import GenderPredictor  # noqa: E402
from gender_predict.data.datasets import ImprovedNameGenderDataset  # noqa: E402


def _datasets(prob, mode="train"):
    p = GenderPredictor(ROOT / "models" / "production", device="cpu")
    p.load()
    df = pd.DataFrame({"primaryName": ["loris busi", "maria de la cruz", "cher", "anna rossi"] * 100,
                       "gender": ["M", "W", "W", "W"] * 100})
    kw = dict(feature_extractor=p.feature_extractor, mode=mode)
    decode = {v: k for k, v in p.preprocessor.char_to_idx.items()}
    text = lambda t: "".join(decode.get(int(i), "") for i in t if int(i))  # noqa: E731
    return (ImprovedNameGenderDataset(df, p.preprocessor, order_swap_prob=0.0, **kw),
            ImprovedNameGenderDataset(df, p.preprocessor, order_swap_prob=prob, **kw), text)


def test_order_swap_only_two_word_names_same_target():
    np.random.seed(0)
    plain, swapped, text = _datasets(0.5)
    changed = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(400):
        a, b = plain[i], swapped[i]
        assert torch.equal(a["gender"], b["gender"])
        if not torch.equal(a["first_name"], b["first_name"]):
            changed[i % 4] += 1
            assert text(b["first_name"]) in ("busi loris", "rossi anna")
    assert changed[1] == 0 and changed[2] == 0          # three or more words, one word: never swapped
    assert 25 <= changed[0] <= 75 and 25 <= changed[3] <= 75   # about half of the two-word names


def test_order_swap_off_outside_training():
    _, val, _ = _datasets(0.5, mode="val")
    assert val.order_swap_prob == 0.0
