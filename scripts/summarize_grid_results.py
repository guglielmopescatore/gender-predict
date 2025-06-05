"""summarise_grid_results.py
-----------------------------------------------------------------
Creates a compact CSV + Markdown table for the **most‑recent N** experiment
folders.  Useful when your `./experiments/` directory contains many old runs
but you want to compare only the latest grid.

Usage
-----
```bash
python summarize_grid_results.py \
       --exp_dir ./experiments \
       --n_last 12               # ← keep only the last N (default 12) \
       --out_csv grid_metrics.csv \
       --out_md  grid_metrics.md
```
The script expects each experiment folder to contain:
* `parameters.json`  – CLI hyper‑parameters
* `logs/test_metrics.json` – at least `{"accuracy":…, "f1":…}`
* optional `logs/val_threshold.json` with `{"best_threshold": …}`

The Markdown table reproduces the same schema you requested:
| ID | Mixup α | Loss | γ | Dropout | TTA n_aug | Threshold | F1 | Accuracy |

-----------------------------------------------------------------"""
import argparse, json, os, re, sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Summarise latest N experiments")
    p.add_argument("--exp_dir", default="./experiments", help="Root folder containing experiment sub‑dirs")
    p.add_argument("--n_last", type=int, default=12, help="How many most‑recent experiments to include")
    p.add_argument("--out_csv", default="grid_metrics.csv")
    p.add_argument("--out_md", default="grid_metrics.md")
    return p.parse_args()

def natural_key(path: Path):
    """Sort by timestamp prefix if present, else mtime."""
    m = re.match(r"(\d{8}_\d{6})_", path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)

def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def main():
    args = parse_args()
    exp_root = Path(args.exp_dir)
    if not exp_root.is_dir():
        sys.exit(f"❌  Directory {exp_root} not found")

    # sort sub‑folders by natural_key desc → pick last N
    folders = sorted([p for p in exp_root.iterdir() if p.is_dir()],
                     key=natural_key, reverse=True)[: args.n_last]

    rows = []
    for p in folders:
        params   = load_json(p / "parameters.json")
        test_met = load_json(p / "logs" / "test_metrics.json")
        thr_file = p / "logs" / "val_threshold.json"
        val_thr  = load_json(thr_file).get("best_threshold", 0.50) if thr_file.exists() else 0.50

        row = {
            "ID"        : params.get("note", p.name)[:6],
            "Mixup α"   : params.get("mixup_alpha", 0.0),
            "Loss"      : params.get("loss", "bce"),
            "γ"         : params.get("gamma", "–"),
            "Dropout"   : params.get("dropout", 0.0),
            "TTA n_aug": params.get("tta_n_aug", 0 if not params.get("use_tta") else params.get("tta_n_aug", "smart" if params.get("tta_strategy") == "smart" else 0)),
            "Threshold" : val_thr,
            "F1"        : test_met.get("f1", "n/a"),
            "Accuracy"  : test_met.get("accuracy", "n/a"),
            "Folder"    : p.name,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # Markdown
    md_lines = ["| ID | Mixup α | Loss | γ | Dropout | TTA n_aug | Threshold | F1 | Accuracy |",
                "|----|---------|------|---|---------|-----------|-----------|----|----------|"]
    for r in rows:
        md_lines.append(
            f"| {r['ID']} | {r['Mixup α']} | {r['Loss']} | {r['γ']} | {r['Dropout']} | "
            f"{r['TTA n_aug']} | {r['Threshold']:.2f} | {r['F1']:.4f} | {r['Accuracy']:.4f} |")
    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"✅  Saved CSV → {args.out_csv} and Markdown → {args.out_md}")

if __name__ == "__main__":
    main()
