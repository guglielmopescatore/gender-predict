"""
Loss functions for gender prediction models.

This module contains various loss functions optimized for binary gender classification,
including Focal Loss and Label Smoothing implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    α  : weight for the positive class (targets == 1)
    γ  : focusing parameter
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        assert reduction in {"mean", "sum", "none"}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : raw model outputs (any real number)
        targets : {0,1} float tensor, same shape
        """
        # Assicurati che i target siano in formato float
        if targets.dtype != torch.float32:
            targets = targets.float()

        # Probabilità sigmoid in modo **sempre** esplicito e numericamente stabile
        prob_pos = torch.sigmoid(logits)
        prob_neg = 1.0 - prob_pos

        # MODIFICA: usa soglia 0.5 per determinare classe positiva/negativa
        # ma mantieni i valori originali di target per la loss
        # pt = torch.where(targets == 1, prob_pos, prob_neg)
        # alpha_t = torch.where(targets == 1,
        #                      torch.full_like(targets, self.alpha),
        #                      torch.full_like(targets, 1 - self.alpha))

        # Nuova versione che funziona con label smoothing
        pt = torch.where(targets >= 0.5, prob_pos, prob_neg)
        alpha_t = torch.where(targets >= 0.5,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1 - self.alpha))

        # BCE senza riduzione + fattore focal
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * ce_loss

        if   self.reduction == "mean": return focal.mean()
        elif self.reduction == "sum" : return focal.sum()
        else                         : return focal


# ------------------------------------------------------------
#  Label‑Smoothing wrapper (facoltativo)
# ------------------------------------------------------------
class LabelSmoothing(nn.Module):
    """
    Avvolge una loss binaria (es. BCE o FocalLoss) e applica smoothing
    al target:  y  →  (1‑ε)·y + ε·0.5
    """
    def __init__(self, base_loss: nn.Module, epsilon: float = 0.05):
        super().__init__()
        self.base_loss = base_loss
        self.epsilon   = epsilon
        assert 0 <= epsilon < 1, "epsilon deve essere tra 0 e 1"

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Assicurati che i target siano in formato float
        if targets.dtype != torch.float32:
            targets = targets.float()

        smooth_targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return self.base_loss(inputs, smooth_targets)


class FocalLossImproved(nn.Module):
    """
    Versione migliorata della Focal Loss con adaptive weighting.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', auto_weight=True):
        super(FocalLossImproved, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.auto_weight = auto_weight
        self.eps = 1e-7
        
    def forward(self, inputs, targets, sample_weights=None):
        """
        inputs: logits dal modello
        targets: target binari (0 o 1)
        sample_weights: pesi per campione (opzionale)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calcola pt
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calcola alpha_t
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            # Auto-weight basato sulla distribuzione del batch
            if self.auto_weight:
                pos_ratio = targets.mean()
                alpha_t = (1 - pos_ratio) * targets + pos_ratio * (1 - targets)
            else:
                alpha_t = 1.0
                
        # Focal loss
        focal_weight = (1 - p_t + self.eps) ** self.gamma
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Applica sample weights se forniti
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
