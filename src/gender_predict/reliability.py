"""
Reliability of a single prediction: "high", "medium" or "low".

Errors are not spread evenly. They concentrate on first names that the model has rarely or never seen, on names
borne by both genders, and on answers close to the decision threshold. A model can ship a table of the first names
of its training file (``first_names.json.gz`` in the model folder: name -> persons, women); with it the predictor
labels each prediction:

  high    the country table decided (see gender_predict.country); or the answer is far from the threshold
          (margin >= HIGH_MARGIN), the first name was seen at least MIN_SEEN times in training and is not ambiguous
          there (less than 10% or more than 90% women)
  low     the answer is close to the threshold (margin < LOW_MARGIN), or the given name was only an initial
  medium  everything else, including names in Han characters and names of which only a part could be read

The margin is 0 at the decision threshold and 1 at probability 0 or 1.

Measured for the model of November 2026 on eight evaluation sets (share of persons; error):
  high    40-89% of persons; error 0.0-0.3% (0.6% and 1.0% on the two sets of non-Western names)
  medium  10-45%; error 2.5-8.7%
  low     1-16%; error 29-36%, with 39-70% of all the errors
The levels describe single predictions; for shares in a population use the error rates by gender instead.
"""

import gzip
import json
from typing import Dict, Optional

HIGH_MARGIN = 0.9
LOW_MARGIN = 0.6
MIN_SEEN = 10
AMBIGUOUS = (0.1, 0.9)


def load_name_table(path) -> Dict[str, list]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["names"]


def margin(probability_female: float, threshold: float) -> float:
    """Distance of the answer from the threshold, rescaled to 0-1 on each side."""
    if probability_female >= threshold:
        return (probability_female - threshold) / (1 - threshold)
    return (threshold - probability_female) / threshold


def reliability_level(probability_female: float, threshold: float, first_name: str, table: Dict[str, list], *,
                      status: str = "ok", han: bool = False, country_table: bool = False) -> Optional[str]:
    """Level for one prediction; None when there is no prediction."""
    if probability_female != probability_female:  # NaN: undetermined
        return None
    if country_table:
        return "high"
    m = margin(probability_female, threshold)
    if status == "initials_dropped" or m < LOW_MARGIN:
        return "low"
    if han or status != "ok" or m < HIGH_MARGIN:
        return "medium"
    seen = table.get(first_name)
    if not seen or seen[0] < MIN_SEEN:
        return "medium"
    share = seen[1] / seen[0]
    return "medium" if AMBIGUOUS[0] < share < AMBIGUOUS[1] else "high"
