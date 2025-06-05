"""calc_thresholds.py
-----------------------------------------------------------------
Find the F1‑optimal threshold on the validation CSV for the latest N
experiments and write `logs/val_threshold.json`.

Assumptions
-----------
* Each experiment folder contains either
  - `logs/val_probs_labels.csv`  (preferred)
  - or `logs/error_analysis.csv` (columns: prob,label)
* CSV must have two columns: one for probability, one for true label.
  The script auto‑detects common aliases:
  - prob  | probability | pred_prob | probas
  - label | target | truth | label_id

Usage
-----
```bash
python calc_thresholds.py \
       --exp_dir ./experiments \
       --n_last 12               # default 12
```
Writes/overwrites `logs/val_threshold.json` with e.g. `{ "best_threshold": 0.46 }`
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def parse_args():
    p = argparse.ArgumentParser(description="Calibrate validation threshold for latest N experiments")
    p.add_argument("--exp_dir", default="./experiments", help="Root folder of experiment sub‑dirs")
    p.add_argument("--n_last", type=int, default=12, help="How many recent folders to process")
    return p.parse_args()

PROB_ALIASES   = {"prob", "probability", "pred_prob", "probas"}
LABEL_ALIASES  = {"label", "target", "truth", "label_id"}
CSV_CANDIDATES = ("val_probs_labels.csv", "error_analysis.csv")


def find_csv(folder: Path):
    for name in CSV_CANDIDATES:
        path = folder / "logs" / name
        if path.exists():
            return path
    return None


def best_threshold(csv_path: Path):
    df = pd.read_csv(csv_path)

    # auto‑detect columns
    prob_col  = next((c for c in df.columns
                    if c.lower() in (
                        "prob", "probability", "pred_prob",
                        "probability_female"          #  ← AGGIUNGI
                    )), None)
    label_col = next((c for c in df.columns
                    if c.lower() in (
                        "label", "label_id", "target", "true_label"  # se serve
                    )), None)

    if not prob_col or not label_col:
        raise ValueError("CSV must contain 'prob' and 'label' columns (or common aliases)")

    y = df[label_col].values
    p = df[prob_col].values

    best_f1, best_thr = 0.0, 0.5
    for t in np.arange(0.30, 0.701, 0.01):
        f1 = f1_score(y, p >= t)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1


def natural_sort_key(p: Path):
    try:
        return p.stat().st_mtime
    except Exception:
        return 0


def main():
    args = parse_args()
    root = Path(args.exp_dir)
    if not root.is_dir():
        sys.exit(f"❌  {root} not found")

    folders = sorted([d for d in root.iterdir() if d.is_dir()],
                     key=natural_sort_key, reverse=True)[: args.n_last]

    for fld in folders:
        csv_path = find_csv(fld)
        if not csv_path:
            print(f"⚠  {fld.name}: validation CSV not found – skipped")
            continue
        try:
            thr, f1 = best_threshold(csv_path)
        except ValueError as e:
            print(f"⚠  {fld.name}: {e} – skipped")
            continue

        (fld / "logs").mkdir(exist_ok=True, parents=True)
        (fld / "logs" / "val_threshold.json").write_text(
            json.dumps({"best_threshold": round(float(thr), 4)}, indent=2)
        )
        print(f"✅  {fld.name}: F1 {f1:.4f} @ thr {thr:.2f}")

if __name__ == "__main__":
    main()
