#!/usr/bin/env python3
"""
Step 2 della migrazione: Modelli e architetture
Estrae e organizza i modelli dai file principali.
"""

import os
import re
from pathlib import Path

def create_file_with_content(path: str, content: str):
    """Crea un file con il contenuto specificato."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def extract_class_from_file(file_path: str, class_name: str) -> str:
    """Estrae una classe specifica da un file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern per trovare la classe
    pattern = rf'class {class_name}.*?(?=\nclass|\n\ndef|\n\nif|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(0)
    else:
        return None

def extract_attention_layer():
    """Estrae AttentionLayer dal file principale."""
    
    with open('train_name_gender_model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Estrai AttentionLayer
    attention_class = extract_class_from_file('train_name_gender_model.py', 'AttentionLayer')
    
    # Estrai anche ImprovedAttentionLayer da improvements.py se esiste
    improved_attention = extract_class_from_file('experiments_improved/improvements.py', 'ImprovedAttentionLayer')
    
    layers_content = '''"""
Attention layers for neural network models.

This module contains various attention mechanisms used in the gender prediction models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
    
    if attention_class:
        layers_content += attention_class + "\n\n"
    
    if improved_attention:
        layers_content += improved_attention + "\n\n"
    
    create_file_with_content('src/gender_predict/models/layers.py', layers_content)

def extract_base_models():
    """Estrae i modelli base da train_name_gender_model.py."""
    
    with open('train_name_gender_model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Estrai le classi del modello
    gender_predictor = extract_class_from_file('train_name_gender_model.py', 'GenderPredictor')
    gender_predictor_enhanced = extract_class_from_file('train_name_gender_model.py', 'GenderPredictorEnhanced')
    
    base_models_content = '''"""
Base gender prediction models.

This module contains the core model architectures for gender prediction:
- GenderPredictor: Basic BiLSTM model with attention
- GenderPredictorEnhanced: Enhanced version with improved capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import AttentionLayer

'''
    
    if gender_predictor:
        base_models_content += gender_predictor + "\n\n"
    
    if gender_predictor_enhanced:
        base_models_content += gender_predictor_enhanced + "\n\n"
    
    create_file_with_content('src/gender_predict/models/base.py', base_models_content)

def extract_v3_model():
    """Estrae il modello V3 da improvements.py."""
    
    # Estrai NameFeatureExtractor
    feature_extractor = extract_class_from_file('experiments_improved/improvements.py', 'NameFeatureExtractor')
    
    # Estrai GenderPredictorV3
    v3_model = extract_class_from_file('experiments_improved/improvements.py', 'GenderPredictorV3')
    
    v3_content = '''"""
Advanced gender prediction model (V3).

This module contains the most advanced model architecture with:
- Multi-head attention
- Feature engineering
- Advanced linguistic features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ImprovedAttentionLayer

'''
    
    if v3_model:
        v3_content += v3_model + "\n\n"
    
    create_file_with_content('src/gender_predict/models/v3.py', v3_content)
    
    # Crea anche il feature extractor
    if feature_extractor:
        feature_content = '''"""
Feature extraction for names.

This module contains utilities for extracting linguistic features from names
for use with advanced models.
"""

import re
from typing import List, Tuple, Dict

''' + feature_extractor
        
        create_file_with_content('src/gender_predict/data/feature_extraction.py', feature_content)

def extract_preprocessors():
    """Estrae i preprocessors."""
    
    # Preprocessor base
    with open('train_name_gender_model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    name_preprocessor = extract_class_from_file('train_name_gender_model.py', 'NamePreprocessor')
    
    base_preprocessing_content = '''"""
Base name preprocessing utilities.

This module contains the core NamePreprocessor class for converting names
to model-ready format.
"""

import pickle
import pandas as pd
from typing import Dict, List

'''
    
    if name_preprocessor:
        base_preprocessing_content += name_preprocessor
    
    create_file_with_content('src/gender_predict/data/preprocessing.py', base_preprocessing_content)
    
    # Preprocessor migliorato
    improved_preprocessor = extract_class_from_file('experiments_improved/improved_preprocessor.py', 'ImprovedNamePreprocessor')
    
    if improved_preprocessor:
        improved_content = '''"""
Improved name preprocessing utilities.

This module contains advanced preprocessing capabilities including:
- Better handling of compound names
- Diacritic normalization
- Smart name order detection
"""

import unicodedata
from typing import Tuple, List, Optional
import pickle

''' + improved_preprocessor
        
        create_file_with_content('src/gender_predict/data/improved_preprocessing.py', improved_content)

def extract_datasets():
    """Estrae le classi Dataset."""
    
    # Dataset base
    name_gender_dataset = extract_class_from_file('train_name_gender_model.py', 'NameGenderDataset')
    
    # Dataset migliorato
    with open('experiments_improved/train_improved_model_v2.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    improved_dataset = extract_class_from_file('experiments_improved/train_improved_model_v2.py', 'ImprovedNameGenderDataset')
    
    datasets_content = '''"""
Dataset classes for gender prediction.

This module contains PyTorch Dataset classes for loading and preprocessing
gender prediction data.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .preprocessing import NamePreprocessor

'''
    
    if name_gender_dataset:
        datasets_content += name_gender_dataset + "\n\n"
    
    if improved_dataset:
        datasets_content += improved_dataset + "\n\n"
    
    create_file_with_content('src/gender_predict/data/datasets.py', datasets_content)

def create_models_init():
    """Crea __init__.py per models con model factory."""
    
    content = '''"""
Gender prediction models.

This module provides all model architectures for gender prediction.
"""

from .base import GenderPredictor, GenderPredictorEnhanced  
from .v3 import GenderPredictorV3
from .layers import AttentionLayer, ImprovedAttentionLayer

# Model factory
MODEL_REGISTRY = {
    'base': GenderPredictor,
    'enhanced': GenderPredictorEnhanced,
    'v3': GenderPredictorV3
}

def create_model(model_type: str, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('base', 'enhanced', 'v3')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return MODEL_REGISTRY[model_type](**kwargs)

__all__ = [
    'GenderPredictor',
    'GenderPredictorEnhanced', 
    'GenderPredictorV3',
    'AttentionLayer',
    'ImprovedAttentionLayer',
    'create_model',
    'MODEL_REGISTRY'
]
'''
    
    create_file_with_content('src/gender_predict/models/__init__.py', content)

def update_data_init():
    """Aggiorna data/__init__.py."""
    
    content = '''"""
Data handling utilities.
"""

from .preprocessing import NamePreprocessor
from .improved_preprocessing import ImprovedNamePreprocessor  
from .datasets import NameGenderDataset, ImprovedNameGenderDataset
from .feature_extraction import NameFeatureExtractor
from .augmentation import NameAugmenter

__all__ = [
    'NamePreprocessor',
    'ImprovedNamePreprocessor',
    'NameGenderDataset', 
    'ImprovedNameGenderDataset',
    'NameFeatureExtractor',
    'NameAugmenter'
]
'''
    
    create_file_with_content('src/gender_predict/data/__init__.py', content)

def main():
    """Esegui Step 2 della migrazione."""
    
    print("🚀 STEP 2: Migrazione modelli e architetture")
    print("=" * 50)
    
    # Verifica prerequisiti
    if not os.path.exists('train_name_gender_model.py'):
        print("❌ Errore: train_name_gender_model.py non trovato")
        return
        
    if not os.path.exists('experiments_improved/improvements.py'):
        print("❌ Errore: experiments_improved/improvements.py non trovato")
        return
    
    print("\n🧠 Estraendo modelli...")
    extract_attention_layer()
    extract_base_models()
    extract_v3_model()
    
    print("\n📊 Estraendo preprocessors e datasets...")
    extract_preprocessors()
    extract_datasets()
    
    print("\n📦 Creando __init__.py files...")
    create_models_init()
    update_data_init()
    
    print("\n✅ STEP 2 COMPLETATO!")
    print("\nFile migrati:")
    print("  • [AttentionLayer] → src/gender_predict/models/layers.py")
    print("  • [GenderPredictor*] → src/gender_predict/models/base.py")
    print("  • [GenderPredictorV3] → src/gender_predict/models/v3.py")
    print("  • [NamePreprocessor] → src/gender_predict/data/preprocessing.py")
    print("  • [ImprovedNamePreprocessor] → src/gender_predict/data/improved_preprocessing.py")
    print("  • [*Dataset] → src/gender_predict/data/datasets.py")
    print("  • [NameFeatureExtractor] → src/gender_predict/data/feature_extraction.py")
    
    print("\n🔄 Prossimi passi:")
    print("  1. Testa che i modelli si importino e istanzino correttamente")
    print("  2. Esegui Step 3: migrazione degli script di training/evaluation")

if __name__ == "__main__":
    main()
