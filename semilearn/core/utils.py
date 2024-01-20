import torch
import random
import numpy as np


def compute_proba(logits: torch.Tensor):
    return torch.softmax(logits, dim=-1)

def seed_everything(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)