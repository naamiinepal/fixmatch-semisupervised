import torch
from torch import nn
import torch.nn.functional as F


def ce_loss(logits, targets, reduction="none"):
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets)

class CELoss(nn.Module):
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits,targets,reduction)