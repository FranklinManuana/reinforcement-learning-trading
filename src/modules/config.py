import torch
import random
import numpy as np


# cuda setup
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# set seed for reporducability 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False