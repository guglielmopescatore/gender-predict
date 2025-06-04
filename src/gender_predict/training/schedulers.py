"""
Advanced learning rate schedulers.
"""

import numpy as np
import torch

class CosineAnnealingWarmupScheduler:
    """
    Learning rate scheduler con warmup e cosine annealing.
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, max_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def mixup_data(x, y, alpha=0.2, is_embedding_idx=False):
    """
    Implementa mixup augmentation per migliorare la generalizzazione.

    Args:
        x: Input data
        y: Target labels
        alpha: Mixup interpolation coefficient
        is_embedding_idx: Se True, x contiene indici per embeddings (non interpolabili)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    if is_embedding_idx:
        # Per indici di embedding, non possiamo interpolare
        # Restituiamo gli indici originali e facciamo mixup solo sui target
        mixed_x = x  # Mantieni gli indici originali
        y_a, y_b = y, y[index]
        # Per NLP, modifichiamo lambda per fare mixup solo sui target
        return mixed_x, y_a, y_b, lam, index
    else:
        # Mixup standard per dati continui
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


