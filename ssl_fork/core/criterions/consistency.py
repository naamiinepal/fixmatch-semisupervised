import torch
from torch import nn

from ssl_fork.core.criterions.cross_entropy import ce_loss


def consistency_loss(logits, targets, mask=None):
    loss = ce_loss(logits,targets, reduction='none')

    if mask is not None:
        loss = loss * mask
    
    return loss.mean()

class ConsistencyLoss(nn.Module):
    def forward(self, logits, targets, mask=None):
        return consistency_loss(logits, targets, mask)