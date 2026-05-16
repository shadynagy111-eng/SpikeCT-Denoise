"""
src/utils/seed.py
-----------------
Deterministic seed setting for all random number generators.
Must be called before any model instantiation or data loading.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False