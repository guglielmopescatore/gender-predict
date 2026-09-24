"""
Transliteration wrapper for gender prediction model.
Converts non-Latin scripts to romanized form matching IMDb style.
"""

import regex as re
from typing import Optional, Tuple
import unicodedata

# Script-specific imports
try:
    from pypinyin import lazy_pinyin, Style
    PINYIN_AVAILABLE = True
except ImportError:
    PINYIN_AVAILABLE = False

try:
    import fugashi
    from romkan import to_hepburn
    JAPANESE_AVAILABLE = True
    tagger = fugashi.Tagger("-Owakati") if JAPANESE_AVAILABLE else None
except ImportError:
    JAPANESE_AVAILABLE = False
    tagger = None

try:
    from hangul_romanize import Transliter
    from hangul_romanize.rule import academic
    KOREAN_AVAILABLE = True
    ko_trans = Transliter(academic) if KOREAN_AVAILABLE else None
except ImportError:
    KOREAN_AVAILABLE = False
    ko_trans = None

try:
    from transliterate import translit
    CYRILLIC_AVAILABLE = True
except ImportError:
    CYRILLIC_AVAILABLE = False


def detect_script(text: str) -> str:
    """
    Detect the primary script of the input text.

    Returns:
        "ZH" for Chinese (Han characters)
        "JA" for Japanese (Hiragana/Katakana)
        "KO" for Korean (Hangul)
        "CYR" for Cyrillic
        "AR" for Arabic
        "LAT" for Latin (default)
    """
    # Remove spaces and punctuation for better detection
    clean_text = re.sub(r'[\s\p{P}]+', '', text)

    if not clean_text:
        return "LAT"

    # Check scripts in order of specificity
    if re.search(r'\p{Hiragana}|\p{Katakana}', clean_text):
        return "JA"
    elif re.search(r'\p{Han}', clean_text):
        return "ZH"
    elif re.search(r'\p{Hangul}', clean_text):
        return "KO"
    elif re.search(r'\p{Cyrillic}', clean_text):
        return "CYR"
    elif re.search(r'\p{Arabic}', clean_text):
        return "AR"
    else:
        return "LAT"


def zh_to_latin(text: str) -> str:
    """Convert Chinese to pinyin without tones."""
    if not PINYIN_AVAILABLE:
        return text
    return ''.join(lazy_pinyin(text, style=Style.NORMAL, strict=False))


def ja_to_latin(text: str) -> str:
    """Convert Japanese to Hepburn romanization."""
    if not JAPANESE_AVAILABLE or not tagger:
        return text

    try:
        # Parse with fugashi to get kana reading
        tokens = tagger(text)
        kana = ''
        for token in tokens:
            if hasattr(token, 'feature') and hasattr(token.feature, 'kana'):
                kana += token.feature.kana or token.surface
            else:
                kana += token.surface

        # Convert kana to romaji
        return to_hepburn(kana).lower()
    except Exception:
        return text


def ko_to_latin(text: str) -> str:
    """Convert Korean to Revised Romanization."""
    if not KOREAN_AVAILABLE or not ko_trans:
        return text

    try:
        return ko_trans.translit(text).lower()
    except Exception:
        return text


def cyrillic_to_latin(text: str) -> str:
    """Convert Cyrillic to Latin."""
    if not CYRILLIC_AVAILABLE:
        # Fallback: basic mapping for common Cyrillic characters
        cyrillic_map = {
            'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
            'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e',
            'Ё': 'E', 'ё': 'e', 'Ж': 'Zh', 'ж': 'zh', 'З': 'Z', 'з': 'z',
            'И': 'I', 'и': 'i', 'Й': 'Y', 'й': 'y', 'К': 'K', 'к': 'k',
            'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n',
            'О': 'O', 'о': 'o', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r',
            'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u',
            'Ф': 'F', 'ф': 'f', 'Х': 'Kh', 'х': 'kh', 'Ц': 'Ts', 'ц': 'ts',
            'Ч': 'Ch', 'ч': 'ch', 'Ш': 'Sh', 'ш': 'sh', 'Щ': 'Shch', 'щ': 'shch',
            'Ъ': '', 'ъ': '', 'Ы': 'Y', 'ы': 'y', 'Ь': '', 'ь': '',
            'Э': 'E', 'э': 'e', 'Ю': 'Yu', 'ю': 'yu', 'Я': 'Ya', 'я': 'ya'
        }
        result = ''
        for char in text:
            result += cyrillic_map.get(char, char)
        return result.lower()

    try:
        # Use transliterate library for more accurate conversion
        return translit(text, 'ru', reversed=True).lower()
    except Exception:
        return text


def arabic_to_latin(text: str) -> str:
    """Convert Arabic to Latin (basic transliteration)."""
    # For now, return as-is since proper Arabic transliteration is complex
    # In production, you might want to use PyICU or arabictransliterator
    return text


def transliterate_name(name: str) -> Tuple[str, str]:
    """
    Transliterate a name from any script to Latin/romanized form.

    Args:
        name: Input name in any script

    Returns:
        Tuple of (transliterated_name, detected_script)
    """
    if not name:
        return name, "LAT"

    # Detect script
    script = detect_script(name)

    # Apply appropriate transliteration
    if script == "ZH":
        result = zh_to_latin(name)
    elif script == "JA":
        result = ja_to_latin(name)
    elif script == "KO":
        result = ko_to_latin(name)
    elif script == "CYR":
        result = cyrillic_to_latin(name)
    elif script == "AR":
        result = arabic_to_latin(name)
    else:
        result = name

    # Post-process to match IMDb style
    # Remove diacritics, lowercase, remove extra spaces
    result = unicodedata.normalize('NFKD', result)
    result = result.encode('ascii', 'ignore').decode('ascii')
    result = result.lower().strip()
    result = re.sub(r'\s+', ' ', result)
    result = result.replace("'", "")
    return result, script


def preprocess_for_model(name: str) -> str:
    """
    Complete preprocessing pipeline for the gender model.

    This function:
    1. Detects the script
    2. Transliterates if needed
    3. Normalizes to match IMDb training data format
    """
    transliterated, script = transliterate_name(name)

    # Additional normalization if needed
    # Remove common titles, punctuation, etc.
    transliterated = re.sub(r'\b(mr|mrs|ms|dr|prof)\.?\s+', '', transliterated, flags=re.IGNORECASE)
    transliterated = re.sub(r'[^\w\s-]', ' ', transliterated)
    transliterated = re.sub(r'\s+', ' ', transliterated).strip()

    return transliterated


