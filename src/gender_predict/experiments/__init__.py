"""
Experiment management and tracking.
"""

from .manager import ExperimentManager
from .comparison import (
    compare_experiments, 
    compare_bias_metrics, 
    generate_full_report,
    compare_learning_curves
)

__all__ = [
    'ExperimentManager',
    'compare_experiments',
    'compare_bias_metrics', 
    'generate_full_report',
    'compare_learning_curves'
]
