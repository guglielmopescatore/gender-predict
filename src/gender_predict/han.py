"""
Chinese, Japanese and Korean names written in their own script.

The gender of a Chinese given name is carried by its characters; pinyin maps many characters to the same syllable
and loses most of it. For a name written in Han characters the predictor therefore (1) separates family and given
name, (2) gives the network the pinyin "given family", as in its training data, and (3) adds to the network's logit
the evidence of the characters: the sum of their log-odds female/male, from a table built on the Comprehensive
Chinese Name Corpus (resources/han_zh.json; CCNC, GPL-3.0), multiplied by HAN_WEIGHT.

Measured on Wikidata persons with a Chinese label, giving the predictor only the characters (November 2026; error
with the network on the pinyin alone -> with the characters added): China 26.1% -> 19.1%, Taiwan 22.0% -> 14.0%,
Hong Kong 30.9% -> 15.8%. Many real names are not marked for gender, so the error stays high (inside CCNC, which
collects typical names, the same table is wrong 4% of the time). 
Japanese names (kanji and kana, family name first, usually without a space) have no reliable romanisation: the
reading of a name cannot be told from its characters. They are not given to the network; the answer comes from the
last two characters of the name, which carry most of the gender (resources/han_ja.json.gz, from the Japanese labels
of 38,458 Wikidata persons, CC0). Japanese persons of the evaluation sets: 5.0% of errors, against 8.4% of the
network on the romanised name. A name with kana is Japanese; one written only in Han characters is assigned to
Japanese or Chinese by the declared country, when there is one, otherwise by a table of lengths, first characters
and last characters (98-99% right on the evaluation persons).

Korean names in hangul are romanised with a table (Revised Romanization, one syllable at a time, given name with a
hyphen, then the family name: "min-seok kim", the form of the training data), which gives the network what the
romanised name would give it; hangul is phonetic and holds no more than that. The last two syllables of the given
name add a little (resources/hangul_ko.json.gz, Korean labels of 18,233 Wikidata persons, CC0). Korean persons of
the evaluation sets: 12.0% of errors with the romanised name of Wikidata, 11.8% with the table, 10.7% with the
syllables added: most of the error is in the names themselves.
"""

import gzip
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


# ---------------------------------------------------------------------------------------------- Japanese
_JA_WORD = re.compile("^[" + chr(0x3040) + "-" + chr(0x30FF) + _HAN_RANGES + chr(0x3005) + chr(0x30F6) + "]+$")   # kana, kanji, iteration mark
_KANA = re.compile("[" + chr(0x3040) + "-" + chr(0x30FF) + "]")
JAPANESE_COUNTRIES = {"JP"}
CHINESE_COUNTRIES = {"CN", "TW", "HK", "MO", "SG"}


@lru_cache(maxsize=1)
def _table_ja():
    with gzip.open(Path(__file__).parent / "resources" / "han_ja.json.gz", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["gender"], data["language"]


def native_name(text: str) -> Optional[str]:
    """The name without brackets and spaces when it is written only in kanji and kana (2 to 8 characters); else None."""
    s = "".join(_BRACKETS.sub("", text).split())
    return s if 2 <= len(s) <= 8 and _JA_WORD.match(s) else None


def is_japanese(name: str, country: str = "") -> bool:
    """Whether a name returned by native_name is Japanese rather than Chinese."""
    if _KANA.search(name):
        return True
    if country in JAPANESE_COUNTRIES:
        return True
    if country in CHINESE_COUNTRIES:
        return False
    lang = _table_ja()[1]
    score = lang.get("len", {}).get(str(min(len(name), 6)), 0.0) + lang.get("f1", {}).get(name[0], 0.0) \
        + lang.get("f2", {}).get(name[:2], 0.0) + lang.get("e1", {}).get(name[-1], 0.0)
    return score >= 0


def japanese_logodds(name: str) -> float:
    """Log-odds female/male from the last and the last-but-one character of a Japanese full name."""
    gender = _table_ja()[0]
    s = gender["l1"].get(name[-1], 0.0)
    if len(name) >= 3:
        s += gender["l2"].get(name[-2], 0.0)
    return s


# ---------------------------------------------------------------------------------------------- Korean
_HANGUL_WORD = re.compile("^[" + chr(0xAC00) + "-" + chr(0xD7A3) + "]+$")
_INITIALS = "g kk n d tt r m b pp s ss  j jj ch k t p h".split(" ")
_MEDIALS = "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split()
_FINALS = " k k k n n n t l k m l l l p l m p p t t ng t t k t p t".split(" ")
# usual spellings of the most frequent family names (elsewhere the syllable is romanised by the table)
KOREAN_FAMILY_NAMES = {"김": "kim", "이": "lee", "박": "park", "최": "choi", "정": "jung", "강": "kang", "조": "cho", "윤": "yoon", "장": "jang",
                       "임": "lim", "신": "shin", "유": "yoo", "안": "ahn", "오": "oh", "권": "kwon", "노": "roh", "백": "baek", "문": "moon", "우": "woo"}
KOREAN_COMPOUND_FAMILY_NAMES = {"남궁", "황보", "제갈", "사공", "선우", "서문", "독고", "동방"}


@lru_cache(maxsize=1)
def _table_ko():
    with gzip.open(Path(__file__).parent / "resources" / "hangul_ko.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)["gender"]


def hangul_name(text: str) -> Optional[str]:
    """The name without brackets and spaces when it is written only in hangul (2 to 4 syllables); else None."""
    s = "".join(_BRACKETS.sub("", text).split())
    return s if 2 <= len(s) <= 4 and _HANGUL_WORD.match(s) else None


def split_korean_name(name: str) -> Tuple[str, str]:
    """(family name, given name): the first syllable, or two for a compound family name; a two-syllable name whose
    first syllable is not a frequent family name is taken as a given name alone."""
    if len(name) >= 3 and name[:2] in KOREAN_COMPOUND_FAMILY_NAMES:
        return name[:2], name[2:]
    if len(name) == 2 and name[0] not in KOREAN_FAMILY_NAMES:
        return "", name
    return name[:1], name[1:]


def _romanise_syllable(ch: str) -> str:
    c = ord(ch) - 0xAC00
    return _INITIALS[c // 588] + _MEDIALS[(c % 588) // 28] + _FINALS[c % 28]


def romanise_korean(name: str) -> str:
    """"given family" in Revised Romanization by syllable, e.g. "min-seok kim"."""
    family, given = split_korean_name(name)
    g = "-".join(_romanise_syllable(c) for c in given)
    f = KOREAN_FAMILY_NAMES.get(family) or "".join(_romanise_syllable(c) for c in family)
    return (g + " " + f).strip()


def korean_logodds(given: str) -> float:
    """Log-odds female/male from the last and the last-but-one syllable of a Korean given name."""
    table = _table_ko()
    s = table["l1"].get(given[-1], 0.0)
    if len(given) >= 2:
        s += table["l2"].get(given[-2], 0.0)
    return s
