"""
Name normalisation, version 2: one function for building training data and for inference.

``normalize_name`` turns a raw full name into the string the model sees (lower-case a-z, hyphen, apostrophe,
space), split into the first token (given name) and the rest (family name and any middle names). Nothing is
dropped silently: letters that Unicode cannot decompose are mapped explicitly, and a name that cannot be
represented (given name in an unsupported script, empty, only an initial for the given name) is returned
with a status other than "ok"/"partial" so that the caller can answer "undetermined" instead of predicting on an empty string.

The function is idempotent on its own "ok" output: normalising twice gives the same result.

Declared limits: the first token is taken as the given name; an all-caps "SURNAME NAME" string cannot be told
from "NAME SURNAME" and is not reordered (only "Surname, Name" with a comma is).
"""

import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple

NORMALIZATION_VERSION = 2
PREDICTABLE = ("ok", "partial", "initials_dropped")  # statuses for which the network is asked
CHARACTERS = "characters"                               # decided from the characters alone (Japanese), no text for the network

# Latin letters with no canonical decomposition (NFKD leaves them untouched).
SPECIAL_LETTERS = {
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "ß": "ss", "ẞ": "SS",
    "ı": "i", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th", "ħ": "h", "Ħ": "H",
    "ŋ": "n", "Ŋ": "N", "ĸ": "k", "ŧ": "t", "Ŧ": "T", "ə": "e", "Ə": "E", "ɛ": "e", "Ɛ": "E", "ɔ": "o", "Ɔ": "O",
    "ƒ": "f", "ɓ": "b", "Ɓ": "B", "ɗ": "d", "Ɗ": "D", "ƙ": "k", "Ƙ": "K", "ƴ": "y", "Ƴ": "Y",
}
APOSTROPHES = "’‘ʼʻ`´′ʹʿʾ"
DASHES = "‐‑‒–—−"
_TRANS = str.maketrans({**SPECIAL_LETTERS, **{c: "'" for c in APOSTROPHES}, **{c: "-" for c in DASHES}})

# Titles: removed at the start of the name. Those in TITLES_DOT only when written with a full stop.
TITLES_BARE = {"dr", "mr", "mrs", "ms", "prof"}
TITLES_DOT = TITLES_BARE | {"dott", "dott.ssa", "prof.ssa", "sig", "sig.ra", "sig.na", "ing", "avv", "on", "rev", "mme", "mlle", "sr", "sra",
                            "srta", "arch", "geom", "rag", "sen", "hon", "mag", "dipl", "dra"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}

_QUOTED = re.compile(r"""\s["“”«»„][^"“”«»„]*["“”«»„](?=\s)|\s'[^']+'(?=\s)|\s\([^)]*\)""")
_NOT_ALLOWED = re.compile(r"[^a-z' -]+")
_HAN = re.compile(r"[⺀-⿟㐀-䶿一-鿿豈-﫿]")
_KANA = re.compile(r"[぀-ヿㇰ-ㇿ]")
_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]")
_CYRILLIC = re.compile(r"[Ѐ-ԯ]")
_GREEK = re.compile("[" + chr(0x370) + "-" + chr(0x3ff) + chr(0x1f00) + "-" + chr(0x1fff) + "]")

# Greek: romanisation in the style of ELOT 743 (the standard of Greek passports), with a table and no external library.
_GREEK_LETTERS = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω", "a v g d e z i th i k l m n x o p r s s t y f ch ps o".split()))
_GREEK_VOICELESS = set("θκξπστφχψ")


def greek_to_latin(text: str) -> str:
    """Romanise the Greek letters of ``text`` (lower case, accents removed); other characters are left as they are."""
    out = []
    for word in text.split():
        w = "".join(c for c in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(c))
        i, r = 0, []
        while i < len(w):
            c, n = w[i], w[i + 1] if i + 1 < len(w) else ""
            if c not in _GREEK_LETTERS:
                r.append(c); i += 1
            elif c == "ο" and n == "υ":
                r.append("ou"); i += 2
            elif c in "αεη" and n == "υ":   # av/af, ev/ef, iv/if: f before a voiceless consonant or at the end
                nn = w[i + 2] if i + 2 < len(w) else ""
                r.append(_GREEK_LETTERS[c] + ("f" if (nn in _GREEK_VOICELESS or nn == "") else "v")); i += 2
            elif c == "μ" and n == "π":
                r.append("b" if i == 0 else "mp"); i += 2
            elif c == "γ" and n and n in "γκχ":
                r.append({"γ": "ng", "κ": "gk", "χ": "nch"}[n]); i += 2
            else:
                r.append(_GREEK_LETTERS[c]); i += 1
        out.append("".join(r))
    return " ".join(out)


class NormalizedName(NamedTuple):
    text: str      # "first rest" (or "first"); empty unless status is "ok" or "partial"
    first: str     # given name: the first token
    rest: str      # everything after the first token
    status: str    # "ok", "partial" (some words unreadable, given name readable), "initials_dropped" (see ``initials``),
                   # "empty", "unsupported_script", "initial_only"
    script: str    # "LAT", "GRE", "ZH", "JA", "KO", "CYR" or "OTHER"
    han: str = ""  # Chinese (script "ZH") and Korean ("KO"): given name in its own script; Japanese (status "characters"):
                   # the whole name (see gender_predict.han)


def detect_script(text: str) -> str:
    if _KANA.search(text):
        return "JA"
    if _HAN.search(text):
        return "ZH"
    if _HANGUL.search(text):
        return "KO"
    if _CYRILLIC.search(text):
        return "CYR"
    return "LAT"


def _romanise(text: str, script: str):
    """Romanise a supported non-Latin script with the same libraries as version 1; None if not possible."""
    try:
        from .inference import transliteration as t
    except Exception:  # noqa: BLE001
        return None
    ok = {"ZH": t.PINYIN_AVAILABLE, "JA": t.JAPANESE_AVAILABLE, "KO": t.KOREAN_AVAILABLE, "CYR": True}[script]
    if not ok:
        return None
    out = {"ZH": t.zh_to_latin, "JA": t.ja_to_latin, "KO": t.ko_to_latin, "CYR": t.cyrillic_to_latin}[script](text)
    return out.replace("'", "") if script == "CYR" else out  # soft signs are rendered as apostrophes


def _fold(text: str) -> str:
    """Explicit letter table, apostrophe and dash variants, diacritics removed, lower case."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.translate(_TRANS).lower()  # after decomposition: a letter such as "ae with macron" is mapped too


@lru_cache(maxsize=400_000)
def normalize_name(name, initials: str = "undetermined", country: str = "") -> NormalizedName:
    """``initials``: what to do when the given name is only an initial ("J. Smith", "S. Radhakrishnan").
    "undetermined" (default): status "initial_only", no text. "drop": the initials are removed and the words that
    follow are the name ("smith", "radhakrishnan"), status "initials_dropped".
    ``country`` (ISO code, optional) is used only to tell Japanese from Chinese names written in Han characters."""
    if not isinstance(name, str):
        name = "" if name is None else str(name)
    text = unicodedata.normalize("NFC", name).strip()
    if not text:
        return NormalizedName("", "", "", "empty", "LAT")
    script, han = "LAT", ""
    if not text.isascii():
        from .han import hangul_name, is_japanese, native_name, romanise_korean, split_han_name, split_korean_name
        korean = hangul_name(text)
        if korean:   # hangul: romanised by the table ("given family"); the given name is kept for its syllables
            text, han, script = romanise_korean(korean), split_korean_name(korean)[1], "KO"
        native = native_name(text)
        if native and is_japanese(native, country):   # Japanese: no text for the network, the characters decide
            return NormalizedName("", "", "", CHARACTERS, "JA", native)
        parts = split_han_name(text)
        if parts is not None:   # written in Han characters: pinyin "given family" for the network, characters kept
            family, han = parts
            given_p, family_p = _romanise(han, "ZH"), _romanise(family, "ZH") if family else ""
            if given_p is None or family_p is None:
                return NormalizedName("", "", "", "unsupported_script", "ZH")
            text = (given_p + " " + family_p).strip()
        if _GREEK.search(text):
            text, script = greek_to_latin(text), "GRE"
        other = detect_script(text)
        if other != "LAT":
            text, script = _romanise(text, other), other
            if text is None:
                return NormalizedName("", "", "", "unsupported_script", script)
    # nicknames in quotes or brackets between name and surname; "Surname, Name"
    text = _QUOTED.sub(" ", " " + text + " ").strip()
    if text.count(",") == 1:
        left, right = (p.strip() for p in text.split(","))
        if left and right:
            text = left if _fold(right).strip(". ") in SUFFIXES else right + " " + left
    tokens = _fold(text).split()
    # titles at the start, generational/academic suffixes at the end
    while len(tokens) > 1 and (("." in tokens[0] and tokens[0].rstrip(".") in TITLES_DOT) or tokens[0] in TITLES_BARE):
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1].strip(".") in SUFFIXES:
        tokens.pop()
    status = "ok"
    unreadable = [any(not c.isascii() and unicodedata.category(c).startswith("L") for c in tok) for tok in tokens]
    if any(unreadable):
        # words in a script that is not handled: without a readable given name there is nothing to predict from;
        # with one ("Marina" + a family name in Greek letters) the readable words are kept and the result is "partial"
        if unreadable[0]:
            return NormalizedName("", "", "", "unsupported_script", "OTHER" if script == "LAT" else script)
        tokens, status, script = [t for t, u in zip(tokens, unreadable) if not u], "partial", "OTHER" if script == "LAT" else script
    text = " ".join(tokens)
    text = _NOT_ALLOWED.sub(" ", text)
    tokens = [t for t in (tok.strip("'-") for tok in text.split()) if t]
    if not tokens:
        return NormalizedName("", "", "", "empty", script)
    # leading initials: dropped when a longer given name follows before the last token ("j robert oppenheimer")
    while len(tokens) > 2 and len(tokens[0]) == 1:
        tokens.pop(0)
    if len(tokens[0]) == 1:
        tokens = [t for i, t in enumerate(tokens) if len(t) > 1 or any(len(u) > 1 for u in tokens[:i])]  # leading initials off
        if initials != "drop" or not tokens:
            return NormalizedName("", "", "", "initial_only", script)
        status = "initials_dropped"
    first, rest = tokens[0], " ".join(tokens[1:])
    return NormalizedName((first + " " + rest).strip(), first, rest, status, script if (script == "KO" or not han) else "ZH", han)


def fit_field(text: str, max_length: int, keep: str = "head") -> str:
    """Cut ``text`` to ``max_length`` characters. ``keep="tail"`` keeps the last whole words (the family name
    proper, when middle names precede it); a single word longer than the limit is cut at its end."""
    if len(text) <= max_length:
        return text
    if keep == "head":
        return text[:max_length]
    words, out = text.split(), ""
    for w in reversed(words):
        cand = w if not out else w + " " + out
        if len(cand) > max_length:
            break
        out = cand
    return out or words[-1][:max_length]
