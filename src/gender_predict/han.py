"""
Chinese names written in characters.

The gender of a Chinese given name is carried by its characters; pinyin maps many characters to the same syllable
and loses most of it. For a name written in Han characters the predictor therefore (1) separates family and given
name, (2) gives the network the pinyin "given family", as in its training data, and (3) adds to the network's logit
the evidence of the characters: the sum of their log-odds female/male, from a table built on the Comprehensive
Chinese Name Corpus (resources/han_zh.json; CCNC, GPL-3.0), multiplied by HAN_WEIGHT.

Measured on Wikidata persons with a Chinese label, giving the predictor only the characters (November 2026; error
with the network on the pinyin alone -> with the characters added): China 26.1% -> 19.1%, Taiwan 22.0% -> 14.0%,
Hong Kong 30.9% -> 15.8%. Many real names are not marked for gender, so the error stays high (inside CCNC, which
collects typical names, the same table is wrong 4% of the time). Japanese names written only in kanji are not told
apart from Chinese ones and go the same way.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

HAN_WEIGHT = 1.0  # weight of the characters' log-odds added to the network's logit (0.25-1.0 tried: 1.0 is the best on every group)
_HAN_RANGES = chr(0x3400) + "-" + chr(0x4DBF) + chr(0x4E00) + "-" + chr(0x9FFF) + chr(0xF900) + "-" + chr(0xFAFF)
_HAN_WORD = re.compile("^[" + _HAN_RANGES + "]+$")
_BRACKETS = re.compile("[" + chr(0xFF08) + r"(].*?[)" + chr(0xFF09) + "]")


@lru_cache(maxsize=1)
def _table():
    with open(Path(__file__).parent / "resources" / "han_zh.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["logodds"], set(data["compound_surnames"])


def split_han_name(text: str) -> Optional[Tuple[str, str]]:
    """(family name, given name) of a name written only in Han characters, family name first; None otherwise.
    With a space the first word is the family name; without, the first character, or the first two when they are
    a frequent compound family name and at least one character follows."""
    parts = _BRACKETS.sub("", text).split()
    if not parts or not all(_HAN_WORD.match(p) for p in parts):
        return None
    if len(parts) >= 2:
        return parts[0], "".join(parts[1:])
    s = parts[0]
    if len(s) == 1:
        return "", s
    k = 2 if (len(s) >= 3 and s[:2] in _table()[1]) else 1
    return s[:k], s[k:]


def han_logodds(given: str) -> float:
    """Sum of the log-odds female/male of the characters of a given name (0 for characters not in the table)."""
    table = _table()[0]
    return sum(table.get(ch, 0.0) for ch in given)
