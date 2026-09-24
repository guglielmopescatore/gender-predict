"""Inference API: load the production model and predict gender from names."""

from .config import InferenceConfig, default_model_dir
from .predictor import GenderPredictor
from .transliteration import detect_script, transliterate_name

__all__ = ["GenderPredictor", "InferenceConfig", "default_model_dir", "detect_script", "transliterate_name"]
