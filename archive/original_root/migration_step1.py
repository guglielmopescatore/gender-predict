#!/usr/bin/env python3
"""
Step 1 della migrazione: File indipendenti e utility
Migriamo per primi i file con meno dipendenze.
"""

import os
import shutil
from pathlib import Path

def create_file_with_content(path: str, content: str):
    """Crea un file con il contenuto specificato."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {path}")

def migrate_losses():
    """Migra losses.py -> src/gender_predict/training/losses.py"""
    
    # Leggi il file originale
    with open('losses.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aggiungi header del nuovo modulo
    new_content = '''"""
Loss functions for gender prediction models.

This module contains various loss functions optimized for binary gender classification,
including Focal Loss and Label Smoothing implementations.
"""

''' + content
    
    create_file_with_content('src/gender_predict/training/losses.py', new_content)

def migrate_sampler():
    """Migra sampler.py -> src/gender_predict/training/samplers.py"""
    
    with open('sampler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = '''"""
Custom samplers for balanced training.

This module provides samplers that ensure balanced class distribution 
during training, particularly useful for imbalanced datasets.
"""

''' + content
    
    create_file_with_content('src/gender_predict/training/samplers.py', new_content)

def migrate_utils():
    """Migra utils.py -> src/gender_predict/utils/common.py"""
    
    with open('utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = '''"""
Common utility functions.

This module contains utility functions used across the gender prediction project,
including early stopping, visualization helpers, and file operations.
"""

''' + content
    
    create_file_with_content('src/gender_predict/utils/common.py', new_content)

def migrate_data_cleaning():
    """Migra dataset_encoding_fix.py -> src/gender_predict/utils/data_cleaning.py"""
    
    with open('dataset_encoding_fix.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = '''"""
Data cleaning and encoding utilities.

This module provides tools for diagnosing and fixing encoding issues,
cleaning datasets, and validating data quality.
"""

''' + content
    
    create_file_with_content('src/gender_predict/utils/data_cleaning.py', new_content)

def create_training_init():
    """Crea __init__.py per training module con imports convenenti."""
    
    content = '''"""
Training utilities for gender prediction models.
"""

from .losses import FocalLoss, LabelSmoothing, FocalLossImproved
from .samplers import BalancedBatchSampler

__all__ = [
    'FocalLoss',
    'LabelSmoothing', 
    'FocalLossImproved',
    'BalancedBatchSampler'
]
'''
    
    create_file_with_content('src/gender_predict/training/__init__.py', content)

def create_utils_init():
    """Crea __init__.py per utils module."""
    
    content = '''"""
Utility functions and helpers.
"""

from .common import EarlyStopping, ensure_dir, save_metrics_to_csv, plot_confusion_matrix

__all__ = [
    'EarlyStopping',
    'ensure_dir', 
    'save_metrics_to_csv',
    'plot_confusion_matrix'
]
'''
    
    create_file_with_content('src/gender_predict/utils/__init__.py', content)

def migrate_improved_components():
    """Migra componenti da experiments_improved/improvements.py"""
    
    # Leggi il file improvements
    with open('experiments_improved/improvements.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Estrai la classe CosineAnnealingWarmupScheduler
    import re
    
    # Trova la classe CosineAnnealingWarmupScheduler
    scheduler_match = re.search(
        r'class CosineAnnealingWarmupScheduler.*?(?=class|\Z)', 
        content, 
        re.DOTALL
    )
    
    if scheduler_match:
        scheduler_content = '''"""
Advanced learning rate schedulers.
"""

import numpy as np

''' + scheduler_match.group(0)
        
        create_file_with_content('src/gender_predict/training/schedulers.py', scheduler_content)
    
    # Estrai NameAugmenter
    augmenter_match = re.search(
        r'class NameAugmenter.*?(?=class|\Z)',
        content,
        re.DOTALL
    )
    
    if augmenter_match:
        augmenter_content = '''"""
Data augmentation for names.
"""

import random
import numpy as np
from typing import List

''' + augmenter_match.group(0)
        
        create_file_with_content('src/gender_predict/data/augmentation.py', augmenter_content)

def update_training_init_with_schedulers():
    """Aggiorna training/__init__.py per includere schedulers."""
    
    content = '''"""
Training utilities for gender prediction models.
"""

from .losses import FocalLoss, LabelSmoothing, FocalLossImproved
from .samplers import BalancedBatchSampler
from .schedulers import CosineAnnealingWarmupScheduler

__all__ = [
    'FocalLoss',
    'LabelSmoothing', 
    'FocalLossImproved',
    'BalancedBatchSampler',
    'CosineAnnealingWarmupScheduler'
]
'''
    
    create_file_with_content('src/gender_predict/training/__init__.py', content)

def create_data_init():
    """Crea __init__.py per data module."""
    
    content = '''"""
Data handling utilities.
"""

from .augmentation import NameAugmenter

__all__ = [
    'NameAugmenter'
]
'''
    
    create_file_with_content('src/gender_predict/data/__init__.py', content)

def main():
    """Esegui Step 1 della migrazione."""
    
    print("🚀 STEP 1: Migrazione file indipendenti")
    print("=" * 50)
    
    # Verifica che siamo nella directory corretta
    if not os.path.exists('losses.py'):
        print("❌ Errore: Esegui questo script dalla root del progetto")
        return
    
    # Migra i file utility (indipendenti)
    print("\n📦 Migrando utilities...")
    migrate_losses()
    migrate_sampler()
    migrate_utils()
    migrate_data_cleaning()
    
    # Migra componenti da improvements.py
    print("\n📦 Migrando componenti avanzati...")
    migrate_improved_components()
    
    # Crea file __init__.py
    print("\n📦 Creando __init__.py files...")
    create_training_init()
    update_training_init_with_schedulers()
    create_utils_init()
    create_data_init()
    
    print("\n✅ STEP 1 COMPLETATO!")
    print("\nFile migrati:")
    print("  • losses.py → src/gender_predict/training/losses.py")
    print("  • sampler.py → src/gender_predict/training/samplers.py")
    print("  • utils.py → src/gender_predict/utils/common.py")
    print("  • dataset_encoding_fix.py → src/gender_predict/utils/data_cleaning.py")
    print("  • [parte di] improvements.py → src/gender_predict/training/schedulers.py")
    print("  • [parte di] improvements.py → src/gender_predict/data/augmentation.py")
    
    print("\n🔄 Prossimi passi:")
    print("  1. Testa che i nuovi moduli si importino correttamente")
    print("  2. Esegui Step 2: migrazione dei modelli")

if __name__ == "__main__":
    main()
