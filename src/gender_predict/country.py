"""
Optional country of the person.

The same first name can belong to different genders in different countries (Andrea, Simone, Nicola, Jan, Kim...).
When the caller declares the country, the predictor looks the first name up in a table of first names by country
(resources/names_by_country.json.gz: counts of persons by gender, from open sources only: Wikidata, registers of
elected officials and candidates of Italy, France and Brazil, Olympic athletes). If the name is clear-cut there
(at least MIN_PERSONS persons, at least PURITY of one gender) the table decides; otherwise the model's answer stands.

Measured in November 2026 (error without country -> with the declared country): Italian municipal administrators
0.60% -> 0.22%, French elected officials 1.02% -> 0.85%, gold standard (40 countries) 2.91% -> 2.68%; never worse
than the model alone on any evaluation set. A rule that trusts the table also for names that are not clear-cut gains
a little more where the first word is the given name, but fails where names are written family name first
(China, Korea, Vietnam), so it is not used.
"""

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

MIN_PERSONS = 5
PURITY = 0.95


@lru_cache(maxsize=1)
def _table():
    with gzip.open(Path(__file__).parent / "resources" / "names_by_country.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)["countries"]


def countries():
    """ISO 3166-1 alpha-2 codes present in the table."""
    return sorted(_table())


def normalize_country(country) -> str:
    """Upper-case two-letter code, or "" when missing or not a string."""
    if not isinstance(country, str):
        return ""
    return country.strip().upper()


def country_lookup(first_name: str, country: str) -> Optional[Tuple[int, int]]:
    """(persons, women) with this normalised first name in the country, or None when the name is not in the table."""
    hit = _table().get(normalize_country(country), {}).get(first_name)
    return (hit[0], hit[1]) if hit else None


def country_decision(first_name: str, country: str) -> Optional[Tuple[bool, float, int]]:
    """(is_female, probability_female, persons) when the name is clear-cut in the country, else None.
    The probability is the share of women with add-one-half smoothing."""
    hit = country_lookup(first_name, country)
    if not hit or hit[0] < MIN_PERSONS:
        return None
    n, w = hit
    share = w / n
    if PURITY <= share or share <= 1 - PURITY:
        return share >= 0.5, (w + 0.5) / (n + 1), n
    return None
