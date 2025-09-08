import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import umap
import seaborn as sns

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device = {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")


def save_data(features, scores, labels, path):
    data = np.concatenate((features.detach().cpu().numpy(), scores.detach().cpu().numpy()), axis=1)
    df = pd.DataFrame(data)
    df['label'] = labels.detach().cpu().numpy()
    df.to_csv(path, index=False)
    print(f"[INFO] Data saved to {path}")


def parse_digits(task_str):
    """
    task_str: '0-n' where n is the last digit included in the task.
    """
    ranges = task_str.split(",")
    digits = []
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            digits.extend(range(start, end + 1))
        else:
            digits.append(int(r))
    return sorted(set(digits))