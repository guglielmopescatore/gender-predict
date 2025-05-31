"""
Feature extraction for names.

This module contains utilities for extracting linguistic features from names
for use with advanced models.
"""

import re
from typing import List, Tuple, Dict

class NameFeatureExtractor:
    """
    Estrae feature linguistiche avanzate dai nomi per migliorare la predizione.
    """
    
    # Suffissi comuni per genere in diverse lingue
    FEMALE_SUFFIXES = [
        # Italiano
        'a', 'ella', 'etta', 'ina', 'uccia',
        # Spagnolo
        'ez', 'ita', 'uela',
        # Francese
        'ette', 'elle', 'ine',
        # Russo
        'ova', 'eva', 'ina', 'aya',
        # Inglese
        'een', 'lyn', 'leigh',
        # Tedesco
        'chen', 'lein',
    ]
    
    MALE_SUFFIXES = [
        # Italiano
        'o', 'i', 'ino', 'ello', 'uccio',
        # Spagnolo
        'os', 'ez', 'ito',
        # Francese
        'eau', 'ot', 'in',
        # Russo
        'ov', 'ev', 'sky', 'ich',
        # Inglese
        'son', 'ton', 'ley',
        # Tedesco
        'mann', 'stein',
    ]
    
    def __init__(self, max_ngram=3):
        self.max_ngram = max_ngram
        self.suffix_to_idx = self._build_suffix_index()
        
    def _build_suffix_index(self):
        """Costruisce un indice dei suffissi."""
        suffix_to_idx = {'<UNK>': 0}
        idx = 1
        
        for suffix in self.FEMALE_SUFFIXES + self.MALE_SUFFIXES:
            if suffix not in suffix_to_idx:
                suffix_to_idx[suffix] = idx
                idx += 1
                
        return suffix_to_idx
    
    def extract_suffix_features(self, name: str) -> List[int]:
        """Estrae feature basate sui suffissi del nome."""
        name = name.lower().strip()
        
        # Gestisci nomi vuoti
        if not name:
            return [0]  # Unknown suffix
        
        features = []
        
        # Controlla suffissi di lunghezza variabile (2-5 caratteri)
        for length in range(2, min(6, len(name) + 1)):
            suffix = name[-length:]
            if suffix in self.suffix_to_idx:
                features.append(self.suffix_to_idx[suffix])
            
        # Padding o troncamento a lunghezza fissa
        if not features:
            features = [0]  # Unknown suffix
            
        return features[:3]  # Massimo 3 suffissi
    
    def extract_phonetic_features(self, name: str) -> Dict[str, float]:
        """Estrae feature fonetiche dal nome."""
        name = name.lower()
        
        # Gestisci nomi vuoti
        if not name:
            return {
                'ends_with_vowel': 0.0,
                'vowel_ratio': 0.0,
                'has_double_consonant': 0.0,
                'length_normalized': 0.0,
            }
        
        features = {
            'ends_with_vowel': 1.0 if name[-1] in 'aeiou' else 0.0,
            'vowel_ratio': sum(1 for c in name if c in 'aeiou') / max(len(name), 1),
            'has_double_consonant': 1.0 if any(name[i] == name[i+1] and name[i] not in 'aeiou' 
                                              for i in range(len(name)-1)) else 0.0,
            'length_normalized': min(len(name) / 20.0, 1.0),  # Normalizzato a [0, 1]
        }
        
        return features
    
    def extract_ngram_features(self, name: str, n: int = 3) -> List[str]:
        """Estrae n-grammi di caratteri."""
        name = f"^{name.lower()}$"  # Aggiungi marcatori di inizio/fine
        ngrams = []
        
        for i in range(len(name) - n + 1):
            ngrams.append(name[i:i+n])
            
        return ngrams

