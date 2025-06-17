import torch
import random
import numpy as np
from datetime import datetime
import os

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # apply seed to GPU calculations if cuda option available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def model_path_name( episodes, save_dir="output/trained_models"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trained_model_for{episodes}_episodes_{timestamp}.pt"
    return filename