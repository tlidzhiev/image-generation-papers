import os
import random

import numpy as np
import torch


def set_worker_seed(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed % (2**32 - 1))
    os.environ['PYTHONHASHSEED'] = str(seed % (2**32 - 1))
