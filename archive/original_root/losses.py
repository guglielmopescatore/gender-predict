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
