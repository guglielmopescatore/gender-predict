"""
Dataset classes for gender prediction.

This module contains PyTorch Dataset classes for loading and preprocessing
gender prediction data.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .preprocessing import NamePreprocessor

class NameGenderDataset(Dataset):
    """Dataset per l'addestramento del modello di predizione del genere."""

    def __init__(self, dataframe, preprocessor, mode='train'):
        """
        Inizializza il dataset.

        Args:
            dataframe: DataFrame pandas con i dati
            preprocessor: Istanza di NamePreprocessor
            mode: 'train', 'val' o 'test'
        """
        self.df = dataframe
        self.preprocessor = preprocessor
        self.mode = mode

        # Mappa genere a indice
        self.gender_to_idx = {'M': 0, 'W': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Estrai nome e cognome
        name_data = self.preprocessor.preprocess_name(row['primaryName'])

        # Converti in tensori
        first_name_tensor = torch.tensor(name_data['first_name'], dtype=torch.long)
        last_name_tensor = torch.tensor(name_data['last_name'], dtype=torch.long)

        # Prepara l'output
        result = {
            'first_name': first_name_tensor,
            'last_name': last_name_tensor
        }

        # Aggiungi il genere per training/validation
        if 'gender' in row and self.mode != 'predict':
            gender_idx = self.gender_to_idx.get(row['gender'], 0)  # Default a 'M' se mancante
            result['gender'] = torch.tensor(gender_idx, dtype=torch.float)

        # Aggiungi l'ID per il test/predizione
        if 'nconst' in row:
            result['id'] = row['nconst']

        return result


class ImprovedNameGenderDataset(NameGenderDataset):
    """
    Estende il dataset originale con feature engineering e augmentation.
    """
    
    def __init__(self, dataframe, preprocessor, feature_extractor=None, mode='train', 
                 augmenter=None, augment_prob=0.0):
        super().__init__(dataframe, preprocessor, mode)
        self.feature_extractor = feature_extractor
        self.augmenter = augmenter
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        
    def __getitem__(self, idx):
        # Ottieni i dati base dalla classe padre
        base_data = super().__getitem__(idx)
        
        # Se non abbiamo feature extractor, ritorna i dati base
        if not self.feature_extractor:
            return base_data
        
        # Estrai il nome completo
        row = self.df.iloc[idx]
        full_name = row['primaryName']
        
        # Applica augmentation se in training
        if self.augmenter and np.random.random() < self.augment_prob:
            full_name = self.augmenter.augment(full_name)
            # Ri-preprocessa il nome augmentato
            name_data = self.preprocessor.preprocess_name(full_name)
            base_data['first_name'] = torch.tensor(name_data['first_name'], dtype=torch.long)
            base_data['last_name'] = torch.tensor(name_data['last_name'], dtype=torch.long)
        
        # Estrai features linguistiche
        first_name, last_name = self.preprocessor.split_full_name(full_name)
        
        first_suffix = self.feature_extractor.extract_suffix_features(first_name)
        last_suffix = self.feature_extractor.extract_suffix_features(last_name)
        
        # Estrai features fonetiche
        phonetic_first = self.feature_extractor.extract_phonetic_features(first_name)
        phonetic_last = self.feature_extractor.extract_phonetic_features(last_name)
        
        # Combina features fonetiche
        phonetic_features = [
            phonetic_first['ends_with_vowel'],
            phonetic_first['vowel_ratio'],
            phonetic_last['ends_with_vowel'],
            phonetic_last['vowel_ratio']
        ]
        
        # Padding per suffix features
        first_suffix = first_suffix + [0] * (3 - len(first_suffix))
        last_suffix = last_suffix + [0] * (3 - len(last_suffix))
        
        # Aggiungi le nuove features
        base_data['first_suffix'] = torch.tensor(first_suffix[:3], dtype=torch.long)
        base_data['last_suffix'] = torch.tensor(last_suffix[:3], dtype=torch.long)
        base_data['phonetic_features'] = torch.tensor(phonetic_features, dtype=torch.float32)
        
        return base_data


