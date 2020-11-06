# Insipiration: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, with_logits=False, reduce=True, weight=None):
        """
        Implements Focal Loss for binary classification (i.e. assumes targets are one-hot).
        FocalLoss is detailed: https://arxiv.org/pdf/1708.02002.pdf

        :param alpha: Class imbalance term
        :param gamma: Used to change focus on misclassified examples
        :param with_logits: Whether inputs are in logit format or probability format
        :param reduce: Whether to get mean loss
        :param weight: Weight value per class with which to multiply loss; size (c,)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none') if self.with_logits \
            else F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = (-bce_loss).exp()
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        # Need self.weight to be of shape self.inputs.shape[-1]
        if self.weight is not None:
            focal_loss *= self.weight

        return torch.mean(focal_loss) if self.reduce else focal_loss
