import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def set_randomSeed(SEED=11):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
