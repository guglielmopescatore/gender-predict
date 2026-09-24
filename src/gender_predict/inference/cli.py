"""
Command-line interface: ``gender-predict`` (or ``python scripts/final_predictor.py``).

    gender-predict "Maria Rossi"
    gender-predict --input names.csv --output results.csv --name-column primaryName
"""

import argparse
import logging
import sys

from .predictor import GenderPredictor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gender-predict",
        description="Predict gender (W/M) from full names with the gender-predict V4-R1 model.",
    )
    p.add_argument("names", nargs="*", help="one or more full names to classify")
    p.add_argument("--single_name", "--name", dest="single_name", help=argparse.SUPPRESS)
    p.add_argument("--input", "-i", help="input CSV file")
    p.add_argument("--output", "-o", help="output CSV file")
    p.add_argument("--name-column", "--name_column", dest="name_column", default="primaryName",
                   help="column holding the full name (default: primaryName)")
    p.add_argument("--model-dir", help="folder with config.json and weights (default: models/production)")
    p.add_argument("--threshold", type=float, help="override the decision threshold")
    p.add_argument("--no-transliterate", action="store_true",
                   help="skip transliteration/normalisation of names before prediction")
    p.add_argument("--device", default="auto", help="cpu, cuda or auto (default)")
    p.add_argument("-q", "--quiet", action="store_true", help="only print results")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    names = list(args.names)
    if args.single_name:
        names.append(args.single_name)
    if not names and not (args.input and args.output):
        build_parser().print_usage()
        print("error: give one or more names, or --input and --output", file=sys.stderr)
        return 2

    predictor = GenderPredictor(
        args.model_dir,
        threshold=args.threshold,
        transliterate=False if args.no_transliterate else None,
        device=args.device,
    )

    if names:
        for r in predictor.predict_many(names):
            extra = ""
            if r.get("was_transliterated"):
                extra = f"  [{r['detected_script']}: {r['transliterated_name']}]"
            print(f"{r['name']}\t{r['predicted_gender']}\tp(W)={r['probability_female']:.3f}"
                  f"\tconfidence={r['confidence']:.3f}{extra}")

    if args.input and args.output:
        out = predictor.predict_csv(args.input, args.output, name_column=args.name_column)
        if not args.quiet:
            counts = out["predicted_gender"].value_counts()
            summary = ", ".join(f"{g}: {n:,} ({n / len(out) * 100:.1f}%)" for g, n in counts.items())
            print(f"{len(out):,} names -> {args.output}  [{summary}; mean confidence {out['confidence'].mean():.3f}]")
            st = predictor.transliteration_stats()
            if st["transliterated_count"]:
                print(f"transliterated {st['transliterated_count']} names: { {k: v for k, v in st['by_script'].items() if k != 'LAT'} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
