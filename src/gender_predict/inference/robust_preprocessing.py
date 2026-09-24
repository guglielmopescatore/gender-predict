"""
Robust name preprocessing for inference.

Wraps the preprocessor fitted at training time with the cleaning steps
(encoding fixes, control-character removal, Unicode normalisation) that
were found to improve results on real-world data.
"""

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


class ProductionRobustPreprocessor:
    """
    Complete production preprocessor with all optimizations.
    Includes ALL cleaning steps that improved comparison dataset performance.
    """

    def __init__(self, base_preprocessor):
        self.base_preprocessor = base_preprocessor
        self.unicode_map = self._build_unicode_mapping()
        self.stats = {
            'total_processed': 0,
            'unicode_conversions': 0,
            'encoding_fixes': 0,
            'cleaning_applied': 0
        }

    def __getattr__(self, name):
        """Delegate to base preprocessor."""
        return getattr(self.base_preprocessor, name)

    def _build_unicode_mapping(self):
        """Build comprehensive Unicode mapping."""
        return {
            # Latin with diacritics - comprehensive mapping
            'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'ā': 'a', 'ă': 'a', 'ą': 'a',
            'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ē': 'e', 'ė': 'e', 'ę': 'e', 'ě': 'e',
            'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ī': 'i', 'į': 'i', 'ı': 'i',
            'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ō': 'o', 'ő': 'o', 'ø': 'o',
            'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ū': 'u', 'ů': 'u', 'ű': 'u', 'ų': 'u',
            'ý': 'y', 'ÿ': 'y', 'ȳ': 'y',
            'ñ': 'n', 'ň': 'n', 'ń': 'n', 'ņ': 'n',
            'ç': 'c', 'č': 'c', 'ć': 'c', 'ĉ': 'c', 'ċ': 'c',
            'ş': 's', 'š': 's', 'ś': 's', 'ŝ': 's',
            'ž': 'z', 'ź': 'z', 'ż': 'z',
            'ř': 'r', 'ŕ': 'r',
            'ł': 'l', 'ľ': 'l', 'ĺ': 'l', 'ļ': 'l',
            'ď': 'd', 'đ': 'd',
            'ť': 't', 'ţ': 't',
            'ğ': 'g', 'ģ': 'g',
            'ķ': 'k',
            'ß': 'ss',

            # Uppercase variants
            'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Ā': 'A', 'Ă': 'A', 'Ą': 'A',
            'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Ē': 'E', 'Ė': 'E', 'Ę': 'E', 'Ě': 'E',
            'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ī': 'I', 'Į': 'I',
            'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ō': 'O', 'Ő': 'O', 'Ø': 'O',
            'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U', 'Ū': 'U', 'Ů': 'U', 'Ű': 'U', 'Ų': 'U',
            'Ý': 'Y', 'Ÿ': 'Y',
            'Ñ': 'N', 'Ň': 'N', 'Ń': 'N', 'Ņ': 'N',
            'Ç': 'C', 'Č': 'C', 'Ć': 'C', 'Ĉ': 'C', 'Ċ': 'C',
            'Ş': 'S', 'Š': 'S', 'Ś': 'S', 'Ŝ': 'S',
            'Ž': 'Z', 'Ź': 'Z', 'Ż': 'Z',
            'Ř': 'R', 'Ŕ': 'R',
            'Ł': 'L', 'Ľ': 'L', 'Ĺ': 'L', 'Ļ': 'L',
            'Ď': 'D', 'Đ': 'D',
            'Ť': 'T', 'Ţ': 'T',
            'Ğ': 'G', 'Ģ': 'G',
            'Ķ': 'K'
        }

    def fix_encoding_issues(self, text):
        """Fix common encoding corruption issues."""
        if not isinstance(text, str):
            return text

        original_text = text

        # Common UTF-8 -> Latin-1 mistakes
        encoding_fixes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã¢': 'â', 'Ã¤': 'ä', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò',
            'Ã¹': 'ù', 'Ã§': 'ç', 'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã¶': 'ö',
            'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-', 'â€"': '-'
        }

        for corrupted, correct in encoding_fixes.items():
            text = text.replace(corrupted, correct)

        text = re.sub(r'�+', '', text)

        if text != original_text:
            self.stats['encoding_fixes'] += 1

        return text

    def conservative_clean_name(self, name):
        """Apply conservative cleaning (all steps from comparison dataset)."""
        if not isinstance(name, str) or pd.isna(name):
            return ""

        original_name = name

        # 1. Fix encoding issues
        name = self.fix_encoding_issues(name)

        # 2. Remove control characters and null bytes
        name = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', name)

        # 3. Normalize Unicode
        name = unicodedata.normalize('NFD', name)
        name = unicodedata.normalize('NFC', name)

        # 4. Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()

        # 5. Very permissive length check (only extremely long)
        if len(name) > 200:
            name = name[:200]

        if name != original_name:
            self.stats['cleaning_applied'] += 1

        return name

    def normalize_unicode(self, text):
        """Comprehensive Unicode normalization."""
        if not isinstance(text, str):
            return ""

        self.stats['total_processed'] += 1
        result = ""

        for char in text:
            if char in self.unicode_map:
                result += self.unicode_map[char]
                if char != self.unicode_map[char]:
                    self.stats['unicode_conversions'] += 1
            elif ord(char) < 128:  # ASCII character
                result += char
            else:
                # Unicode decomposition fallback
                try:
                    decomposed = unicodedata.normalize('NFD', char)
                    ascii_char = ''.join(c for c in decomposed
                                       if unicodedata.category(c) != 'Mn')

                    if ascii_char and ord(ascii_char[0]) < 128:
                        result += ascii_char
                        self.stats['unicode_conversions'] += 1
                    else:
                        result += '?'  # Conservative fallback
                except Exception:
                    result += '?'

        return result

    def preprocess_name(self, full_name):
        """
        Complete production preprocessing pipeline.
        Applies ALL optimizations found during comparison dataset analysis.
        """
        try:
            # Step 1: Conservative cleaning (encoding, control chars, etc.)
            cleaned_name = self.conservative_clean_name(full_name)

            # Step 2: Unicode normalization
            normalized_name = self.normalize_unicode(cleaned_name)

            # Step 3: Use base preprocessor
            result = self.base_preprocessor.preprocess_name(normalized_name)

            # Add metadata for monitoring
            result['_processing_metadata'] = {
                'original_name': full_name,
                'cleaned_name': cleaned_name,
                'normalized_name': normalized_name,
                'was_processed': full_name != normalized_name,
                'processing_steps': [
                    'conservative_cleaning',
                    'unicode_normalization',
                    'base_preprocessing'
                ]
            }

            return result

        except Exception as e:
            # Emergency fallback
            logger.warning("Preprocessing error for %r: %s", full_name, e)

            # Use base preprocessor directly as fallback
            try:
                return self.base_preprocessor.preprocess_name(str(full_name))
            except:
                # Ultimate fallback
                if hasattr(self.base_preprocessor, 'max_name_length'):
                    max_len = self.base_preprocessor.max_name_length
                    max_surname = self.base_preprocessor.max_surname_length
                else:
                    max_len = max_surname = 20

                return {
                    'first_name': [0] * max_len,
                    'last_name': [0] * max_surname,
                    '_processing_metadata': {
                        'error': str(e),
                        'fallback_used': True
                    }
                }

    def get_processing_stats(self):
        """Get comprehensive processing statistics."""
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['unicode_conversion_rate'] = stats['unicode_conversions'] / stats['total_processed']
            stats['encoding_fix_rate'] = stats['encoding_fixes'] / stats['total_processed']
            stats['cleaning_rate'] = stats['cleaning_applied'] / stats['total_processed']

        return stats

    def split_full_name(self, full_name):
        """
        Split full name into first and last name.
        Simple implementation for compatibility.
        """
        if not isinstance(full_name, str):
            return "", ""
        
        parts = full_name.strip().split()
        if len(parts) == 0:
            return "", ""
        elif len(parts) == 1:
            return parts[0], ""
        else:
            first_name = parts[0]
            last_name = " ".join(parts[1:])
            return first_name, last_name
