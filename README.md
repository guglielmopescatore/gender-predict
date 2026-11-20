# gender-predict

Gender (W/M) from a person's full name, for research on aggregates: a small neural model that runs offline on a CPU,
with a Python API and a command-line tool. Version 4.0 (November 2026).

- **Free, offline, reproducible.** No service to call, no cost per name; the model (18 MB) is in the repository and
  the same name always gets the same answer.
- **Measured.** Error rates are reported separately for women and men, on eight evaluation sets from different
  sources and countries, because a difference between the two rates distorts shares and comparisons.
- **Honest about what it cannot know.** A name that cannot be read is returned as *undetermined*; every prediction
  carries a reliability level; the limits are listed below.

The model works from the name alone. It is meant for counting and comparing groups, not for deciding the gender of
an individual: see [Limits](#limits) and [Ethical considerations](#ethical-considerations).

## Installation

```bash
git clone https://github.com/guglielmopescatore/gender-predict.git
cd gender-predict
pip install -e .
```

Python ≥ 3.8. Inference works right after cloning; no GPU is required.

## Quick start

### Command line

```bash
gender-predict "Maria Rossi" "Søren Jørgensen" "佐藤花子" "김민석" "Γιώργος Παπαδόπουλος"

# a CSV file: adds predicted_gender, probability_female, confidence, status, reliability
gender-predict --input names.csv --output results.csv --name-column name

# with the country of the persons (one for all, or a column with one code per row)
gender-predict --country IT "Andrea Rossi"
gender-predict --input names.csv --output results.csv --name-column name --country-column country
```

Other options: `--threshold`, `--model-dir`, `--device`, `-q`. `python scripts/final_predictor.py ...` works the same
way without installing.

### Python

```python
from gender_predict import GenderPredictor

predictor = GenderPredictor()                      # loads models/production
predictor.predict("Maria Rossi")
# {'name': 'Maria Rossi', 'predicted_gender': 'W', 'probability_female': 0.998, 'confidence': 0.998,
#  'threshold_used': 0.46, 'reliability': 'high', 'status': 'ok', 'transliterated_name': 'maria rossi',
#  'detected_script': 'LAT', 'was_transliterated': False}

predictor.predict("Andrea Rossi", country="IT")    # decided by the table of Italian first names
predictor.predict_many(names, country=countries)   # one country for all names, or one per name (None where unknown)
df = predictor.predict_dataframe(df, name_column="name", country_column="country")
```

| Field | Meaning |
|---|---|
| `predicted_gender` | `W`, `M`, or `Unknown` when the name is undetermined |
| `probability_female` | probability of `W` (NaN when undetermined); `W` at or above the threshold (0.46) |
| `status` | `ok`; `partial` (part of the name was in a script that is not handled, the given name was readable); `characters` (Japanese name, decided from its characters); or the reason why the name is undetermined: `empty`, `initial_only`, `unsupported_script` |
| `reliability` | `high`, `medium` or `low` (see below) |
| `source` | only when a country is given: `country_table` or `model`; with `country_table` the model's own answer is in `model_probability_female` |
| `confidence` | probability of the predicted class |

The name is expected as "given name, then family name". "Family name, Given name" with a comma is reordered; titles
(Dr., Prof., Sig.ra...) and nicknames in quotes are removed. Names written family name first without a comma are
handled as far as the model has learned them (Chinese, Korean and Vietnamese names were given in both orders in
training).

## What the tool does beyond the model

**Country (optional).** The same first name can belong to different genders in different countries (Andrea, Simone,
Nicola, Jan, Kim). When the country is declared and the first name is clear-cut there (at least 5 persons, at least
95% of one gender in a table of first names by country built from open sources), the table decides. This never did
worse than the model alone on any evaluation set; where local name lists are good the gain is large (Italian
administrators: from 0.60% to 0.22% of errors).

**Names in their own script.** Greek, Cyrillic and Korean names are romanised with tables; for Chinese and Japanese
names the gender is in the characters, which romanisation would lose, so the characters are used directly (alone for
Japanese, together with the model for Chinese). Error on Wikidata persons, name in original script against the
same persons' romanised name: Japan 5.3% (8.4%), Korea 10.8% (11.9%), China 18.9% (22.7%), Taiwan 13.8% (18.4%);
Hong Kong 16.1% (10.3%: there the romanised name often holds an English given name). Hebrew, Arabic and the other
scripts are returned as undetermined: a letter-by-letter table gives forms the model has never seen.

**Undetermined.** No answer is given when there is nothing to read: an empty string, a given name reduced to an
initial ("J. Smith"), a given name in a script that is not handled. On the gold standard this is 0.4% of the names.

**Reliability of each prediction.** From the distance of the answer from the threshold and from how often, and how
ambiguously, the first name was seen in training:

| Level | Share of persons | Error |
|---|---|---|
| high | 40–89% | 0.0–0.3% (up to 1.0% on non-Western names) |
| medium | 10–45% | 2.5–8.7% |
| low | 1–16% | 29–36%; more than half of all the errors are here |

(ranges over the eight evaluation sets). With the country declared the share of "high" grows.

## Performance

Production model, error rates in percent on the names it determines. *F→M*: women classified as men; *M→F*: men
classified as women.

| Evaluation set | Persons | Error | F→M | M→F |
|---|---|---|---|---|
| Wikidata persons, 40 countries (gold standard, gender-balanced) | 97,273 | 2.83 | 3.74 | 1.99 |
| – Western countries | 63,179 | 1.28 | 1.63 | 0.96 |
| – other countries | 34,094 | 5.72 | 7.72 | 3.89 |
| – with the country declared | | 2.61 | 3.48 | 1.80 |
| Wikidata persons, 32 further countries | 8,046 | 5.73 | 7.23 | 4.22 |
| Italian local administrators, 2025 | 107,431 | 0.60 | 0.28 | 0.76 |
| – with the country declared | | 0.22 | 0.34 | 0.16 |
| French elected officials | 20,000 | 1.02 | 1.24 | 0.82 |
| Brazilian candidates, 2022 | 4,391 | 1.69 | 1.62 | 1.72 |
| Olympic athletes | 23,043 | 3.11 | 5.52 | 2.18 |
| Santamaría & Mihaljević (2018) benchmark | 5,779 | 4.72 | 6.61 | 3.75 |
| Wikidata persons of 47 non-Western countries, names as in their label | 13,934 | 9.75 | 13.14 | 6.34 |

The number of persons includes the few that are undetermined (0.4% on the gold standard, none in the registers).
The 95% confidence interval of the error on the gold standard is 2.73–2.94. No person of these sets, and no homonym,
was in the training data. Three trainings with different seeds differ by a few hundredths of a point on the gold
standard. With the same procedure the model of version 3.2 (V4-R1) had 3.93% of errors on the gold standard (F→M
5.87, M→F 2.12) and 1.35% on the Italian administrators.

Speed, measured on a desktop computer: about 1,400 names per second on the CPU and 42,000 on a GPU (RTX 4090).

### Using the predictions in aggregate

The two error rates are not equal, so the share of women counted from the predictions is biased, and the bias
changes with the population: it is larger where names are rarer or more ambiguous. On the gold standard, where
48.2% of the persons are women, the predictions count 47.4%; on the names of 47 non-Western countries 46.7%
against 50.2%; on the Italian administrators 34.7% against 34.3%.

- If *a* is the F→M rate and *b* the M→F rate for a population like yours, an observed share *p̂* corresponds to
  *p = (p̂ − b) / (1 − a − b)*. The rates in the table above can be used when your population resembles one of
  those sets; a hand-checked sample of your own data is better.
- **Do not sum `probability_female` to estimate a share.** The probabilities carry the share of women of the
  training data (about 36%): the sum is close to the truth where the true share is near that value (Italian
  administrators: 34.4% against 34.3%) and too low where it is higher (gold standard: 45.4% against 48.2%), worse
  than counting the predictions.
- Declaring the country and checking the names of "low" reliability by hand reduce the error; they do not remove it.
- When comparing groups (cohorts, countries, professions), consider that the error rates may differ between them:
  a difference in error rates can create, hide or inflate a difference between the groups.

## Limits

- **The name is not the person.** About 0.3% of the gold-standard persons bear a name that is clear-cut for the other
  gender; no method based on names can get them right. Non-binary identities are not represented.
- **Rare and ambiguous names.** Half of the remaining errors fall on first names seen fewer than 100 times in
  training (18% of persons), more than a third on names borne by both genders.
- **Chinese names.** In pinyin most of the gender information is lost (23% of errors); even from the characters real
  names are often not marked for gender (19%).
- **Order of the name.** The model expects the given name first ("Loris Busi"); "Busi, Loris" with a comma is
  reordered, "Busi Loris" is not recognised and about one name in four is then wrong (26–28% on our evaluation
  samples, against 0.6–2.7% in the usual order). Reorder such files before prediction: no general rule exists for
  names of more than two words, so this is left to the user, who knows the convention of the source. A model can
  also be trained to accept both orders (see "Training and evaluation").
- **Initials.** A given name reduced to an initial is undetermined, including South Indian names where the initial
  precedes the personal name ("S. Radhakrishnan").
- **Kanji.** A name written only in Han characters is assigned to Japanese or Chinese by the declared country or,
  without it, by a table that is right 98–99% of the time.
- **Training data.** Most of the training names come from IMDb, whose gender labels contain errors (women recorded
  under male roles more often than the reverse) and over-represent Western names.

## Models

- `models/production/` — version 4.0: one character-level BiLSTM with multi-head attention on the whole normalised
  name (4.6 M weights), trained on 6.97 M names (IMDb; Wikidata; registers of elected officials and candidates of
  Italy, France and Brazil; Olympic athletes) and distilled from three models. Details in the folder's README.
- `models/v4r1/` — the production model of versions up to 3.2 (June 2025), kept for reproducibility:
  `GenderPredictor("models/v4r1")` or `--model-dir models/v4r1`.

The tables shipped with the package (`src/gender_predict/resources/`, sources and licences in its README) are built
from open data: Wikidata (CC0), the Comprehensive Chinese Name Corpus (GPL-3.0), public registers of Italy, France
and Brazil, Olympic athletes (CC0). The IMDb data used for training are not redistributed.

## Training and evaluation

`scripts/train_model.py` trains a model (`--help` lists the options); the production model was trained with
`--round 3 --single_string --advanced_preprocessing --augment_version 2 --early_stop 0 --epochs 20` and soft targets
(`--soft_target_column`). `scripts/evaluate_model.py` and `scripts/experiment_tools.py` evaluate and compare
experiments; `tools/` holds utilities for thresholds and batches. The name normalisation used both to build the
training data and at inference is `gender_predict.normalization.normalize_name`.

The training file expects the columns `primaryName` and `gender` (`M`/`W`).

Variants for particular needs can be trained with the same script. As an example, `--order_swap_prob 0.10` shows
the model one two-word name in ten family name first, with the same target. With the recipe of the production model
(20 epochs, about 80 minutes on one RTX 4090, reusing the soft targets) the error on names written family name first
falls from 27.6% to 2.1% on Italian local administrators and from 26.0% to 4.5% on the gold sample, at the cost of
about 0.1 points in the usual order (gold 2.94% against 2.83%); names of more than two words remain weaker. For this
reason the variant is not the default model. A model trained this way is loaded like any other:
`GenderPredictor("<model folder>")` or `--model-dir <model folder>`.

Tests: `pytest tests/` (normalisation, tables, and the regression baselines of both models).

## Repository structure

```
src/gender_predict/     package: inference API, normalisation, tables for countries and scripts, models, training code
models/                 production model (4.0) and previous model (V4-R1)
scripts/                training, evaluation, command-line predictor
tools/, examples/       utilities and sample-data helpers
tests/                  unit and regression tests
```

## Ethical considerations

The tool infers a statistical association between names and a binary gender category; it says nothing certain
about any individual and should not be used to make decisions about individuals. It is intended for research on
aggregates, where its error rates, reported above by gender, can be taken into account. Name–gender associations
vary across cultures and the model is less accurate for names that are under-represented in its training data.
Consider the privacy implications of processing personal data and your institution's ethics guidelines.

## Use of AI tools

Part of the code, of the tests and of the analyses behind this repository was developed with the assistance of a
large language model (Claude, by Anthropic), used as a programming assistant under the direction of the author, who
reviewed and tested the results and remains responsible for the software. Commits made with this assistance carry a
`Co-Authored-By: Claude` line, which is why "Claude" is listed among the contributors on GitHub; the model is a tool,
not an author of the project.

## License

GNU General Public License v3.0 — see the LICENSE file.

## Citation

```bibtex
@software{gender_predict,
  title={gender-predict: gender from names for research on aggregates},
  author={Guglielmo Pescatore},
  year={2026},
  version={4.0.1},
  url={https://github.com/guglielmopescatore/gender-predict}
}
```

## Changelog

- **4.0.1** — Training option `--order_swap_prob` (names shown family name first during training, for models that
  must accept both orders) and documentation of the order of the name. The production model and all predictions
  are unchanged.
- **4.0.0** — New production model (single string, distilled; 2.8% of errors on the gold standard against 3.9%),
  half the size of the previous one. One name normalisation for training and inference. Names that cannot be read
  are returned as undetermined (`predicted_gender` "Unknown", `status`). Greek, Chinese, Japanese and Korean names
  in their own script. Optional country. Reliability level of each prediction. The decision threshold is 0.46.
  The previous model is in `models/v4r1/`; results differ from version 3.2 and are not directly comparable.
- **3.2.0** — Fix: the model is now rebuilt with the number of attention heads it was trained with (8 for V4-R1).
  Earlier versions rebuilt it with 4; the weights loaded without error but the attention was computed differently,
  which changed about 0.3% of the predicted labels. `predict_many` processes names in blocks.
- **3.1.0** — Inference moved into the package (`GenderPredictor`), single `config.json`, optional `[ja]` extra.
