"""
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
