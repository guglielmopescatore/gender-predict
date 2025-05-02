import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.

    Focal Loss reduces the relative loss for well-classified examples, focusing more on hard,
    misclassified examples.

    Args:
        gamma: Focusing parameter (default: 2). Higher gamma = more focus on hard examples.
        alpha: Weighting factor (default: 0.3). Controls weight for class 1 (alpha for class 1, 1-alpha for class 0).
        reduction: Loss reduction method ('mean', 'sum', or 'none').
    """
    def __init__(self, gamma=2, alpha=0.3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities if inputs are logits
        if isinstance(inputs, torch.Tensor) and inputs.shape == targets.shape:
            # Input is already probability (0-1)
            probs = inputs
        else:
            # Input is logit
            probs = torch.sigmoid(inputs)

        # Calculate BCE loss
        BCE_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # Calculate probabilities for the target class
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothing(nn.Module):
    """
    Label smoothing wrapper for binary classification.

    Smooths the target labels to prevent overconfidence and improve generalization.

    Args:
        base_loss: The underlying loss function to wrap
        epsilon: Smoothing factor (default: 0.1). Controls how much to smooth the labels.
    """
    def __init__(self, base_loss, epsilon=0.1):
        super(LabelSmoothing, self).__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Apply label smoothing (move targets closer to 0.5)
        smoothed_targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon

        # Apply the base loss function with smoothed targets
        return self.base_loss(inputs, smoothed_targets)
