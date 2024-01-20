import torch
import random
import numpy as np
from pathlib import Path

def compute_proba(logits: torch.Tensor):
    return torch.softmax(logits, dim=-1)

def seed_everything(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)

def get_latest_checkpoint(path, checkpoint_regex="*.pt"):
    """get latest model checkpoint based on file creation date"""
    checkpoints = list(Path(path).rglob(checkpoint_regex))
    latest_checkpoint_path = max(checkpoints, key=lambda x: x.lstat().st_ctime)
    return str(latest_checkpoint_path)

if __name__ == '__main__':
    ckpt_path = get_latest_checkpoint('checkpoint/semi-supervised/')
    print(ckpt_path)