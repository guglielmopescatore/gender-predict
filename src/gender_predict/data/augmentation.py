"""
Data augmentation for names.
"""

import random
import numpy as np
from typing import List

class NameAugmenter:
    """
    Data augmentation per nomi per migliorare la robustezza del modello.
    """
    
    def __init__(self, augment_prob=0.15, version=1):
        self.augment_prob = augment_prob
        # version 1 (historical): the caller draws with augment_prob and augment() draws again, so the effective
        # rate is augment_prob squared (0.25 -> 6%); typos may insert digits; one variant in four changes only the
        # case, which the preprocessor removes. version 2: augment() always applies one of typo (letters only),
        # duplicate, swap; the caller's draw is the only one, so the declared rate is the effective rate.
        self.version = version
        
        # Mapping per errori di battitura comuni
        self.typo_map = {
            'a': ['q', 's'],
            'e': ['w', 'r', '3'],
            'i': ['u', 'o', '1', 'l'],
            'o': ['i', 'p', '0'],
            'u': ['y', 'i'],
            's': ['z', 'a'],
            'c': ['k', 'x'],
            'k': ['c'],
            'y': ['i'],
            'v': ['b'],
            'b': ['v'],
            'n': ['m'],
            'm': ['n'],
        }
        
    def augment(self, name: str) -> str:
        """Applica augmentation casuale al nome."""
        if not isinstance(name, str):
            return name
        if self.version >= 2:
            augmentation_type = random.choice(['typo', 'duplicate', 'swap'])
        elif random.random() > self.augment_prob:
            return name
        else:
            augmentation_type = random.choice(['typo', 'case', 'duplicate', 'swap'])
        
        if augmentation_type == 'typo':
            return self._add_typo(name)
        elif augmentation_type == 'case':
            return self._random_case(name)
        elif augmentation_type == 'duplicate':
            return self._duplicate_char(name)
        elif augmentation_type == 'swap':
            return self._swap_chars(name)
            
        return name
    
    def _add_typo(self, name: str) -> str:
        """Simula errori di battitura."""
        if len(name) < 2:
            return name
            
        name_list = list(name.lower())
        idx = random.randint(0, len(name_list) - 1)
        char = name_list[idx]
        
        if char in self.typo_map:
            options = [c for c in self.typo_map[char] if not (self.version >= 2 and c.isdigit())]
            name_list[idx] = random.choice(options)
            
        return ''.join(name_list)
    
    def _random_case(self, name: str) -> str:
        """Cambia casualmente maiuscole/minuscole."""
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in name)
    
    def _duplicate_char(self, name: str) -> str:
        """Duplica casualmente un carattere."""
        if len(name) < 2:
            return name
            
        idx = random.randint(0, len(name) - 1)
        return name[:idx] + name[idx] + name[idx:]
    
    def _swap_chars(self, name: str) -> str:
        """Scambia due caratteri adiacenti."""
        if len(name) < 2:
            return name
            
        idx = random.randint(0, len(name) - 2)
        name_list = list(name)
        name_list[idx], name_list[idx + 1] = name_list[idx + 1], name_list[idx]
        
        return ''.join(name_list)


