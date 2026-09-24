"""
Gender Prediction Package
"""

__version__ = "3.1.0"

# Import solo i moduli base che funzionano
from .models import create_model, MODEL_REGISTRY
from .data import NamePreprocessor, NameGenderDataset
from .experiments import ExperimentManager
from .inference import GenderPredictor

__all__ = [
    'create_model',
    'MODEL_REGISTRY', 
    'NamePreprocessor',
    'NameGenderDataset',
    'ExperimentManager',
    'GenderPredictor',
    '__version__'
]
