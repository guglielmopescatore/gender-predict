"""
Base name preprocessing utilities.

This module contains the core NamePreprocessor class for converting names
to model-ready format.
"""

import pickle
import pandas as pd
from typing import Dict, List

class NamePreprocessor:
    """Classe per il preprocessing dei nomi."""

    def __init__(self, max_name_length=20, max_surname_length=20):
        """
        Inizializza il preprocessore.

        Args:
            max_name_length: Lunghezza massima per i nomi
            max_surname_length: Lunghezza massima per i cognomi
        """
        self.max_name_length = max_name_length
        self.max_surname_length = max_surname_length
        self.char_to_idx = {'<PAD>': 0}  # Inizializza con padding token

        # Aggiungi caratteri alfabetici (a-z)
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.char_to_idx[c] = i + 1

        # Aggiungi alcuni caratteri speciali comuni nei nomi
        for i, c in enumerate("'-áàâäãåçéèêëíìîïñóòôöõúùûüýÿ"):
            self.char_to_idx[c] = len(self.char_to_idx)

        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def clean_name(self, name):
        """Normalizza un nome."""
        if not isinstance(name, str) or pd.isna(name):
            return ""
        # Converti in minuscolo e rimuovi spazi iniziali/finali
        name = name.lower().strip()
        # Rimuovi caratteri non alfabetici (tranne quelli speciali)
        name = ''.join(c for c in name if c.isalpha() or c in self.char_to_idx)
        return name

    def split_full_name(self, full_name):
        """Divide un nome completo in nome e cognome."""
        if not isinstance(full_name, str) or pd.isna(full_name):
            return "", ""

        parts = full_name.split()
        if len(parts) == 1:
            return parts[0], ""  # Solo nome, nessun cognome
        elif len(parts) == 2:
            return parts[0], parts[1]  # Caso normale: nome e cognome
        else:
            # Casi complessi: prendiamo il primo come nome, il resto come cognome
            return parts[0], " ".join(parts[1:])

    def name_to_indices(self, name, max_length):
        """Converte un nome in una sequenza di indici."""
        name = self.clean_name(name)
        # Converti ogni carattere in indice
        indices = [self.char_to_idx.get(c, 0) for c in name[:max_length]]
        # Padding
        indices = indices + [0] * (max_length - len(indices))
        return indices

    def preprocess_name(self, full_name):
        """Preprocessa un nome completo."""
        first_name, last_name = self.split_full_name(full_name)

        first_name_indices = self.name_to_indices(first_name, self.max_name_length)
        last_name_indices = self.name_to_indices(last_name, self.max_surname_length)

        return {
            'first_name': first_name_indices,
            'last_name': last_name_indices
        }

    def save(self, path):
        """Salva il preprocessore su disco."""
        with open(path, 'wb') as f:
            pickle.dump({
                'max_name_length': self.max_name_length,
                'max_surname_length': self.max_surname_length,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)

    @classmethod
    def load(cls, path):
        """Carica un preprocessore da disco."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls(
            max_name_length=data['max_name_length'],
            max_surname_length=data['max_surname_length']
        )

        preprocessor.char_to_idx = data['char_to_idx']
        preprocessor.idx_to_char = data['idx_to_char']
        preprocessor.vocab_size = data['vocab_size']

        return preprocessor
