"""
Gender Prediction Package
"""

__version__ = "1.0.0"

# Import solo i moduli base che funzionano
from .models import create_model, MODEL_REGISTRY
from .data import NamePreprocessor, NameGenderDataset
from .experiments import ExperimentManager

__all__ = [
    'create_model',
    'MODEL_REGISTRY', 
    'NamePreprocessor',
    'NameGenderDataset',
    'ExperimentManager',
    '__version__'
]
