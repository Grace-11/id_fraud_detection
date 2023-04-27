import random
import numpy as np
import os
import torch  

def set_all_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def float2timeformat(seconds):
    s = seconds % 60
    m = seconds % 3600 // 60
    h = seconds // 3600
    return "%2d:%2d:%2f" % (h, m, s)


