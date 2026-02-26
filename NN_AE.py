import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import random
import os
import matplotlib.pyplot as plt


# -----------------------------
# Model: encoder -> f_size-> logits, plus decoder
# -----------------------------
class CALM_AE_NN(nn.Module):
    def __init__(self, f_size=64, num_classes=10):
        super().__init__()
        # ---- Encoder ----
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # [B,10,24,24]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # [B,20,20,20]
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.adapter = nn.Linear(160, f_size) # f_size features
         
        # ---- Classifier head ----
        self.fc2 = nn.Linear(f_size, 16) 
        self.fc3 = nn.Linear(16, num_classes)

        # ---- Decoder ----
        # decode from f_size feature to 28×28
        self.fc_dec1 = nn.Linear(f_size, 160)
        self.fc_dec2 = nn.Linear(160, 320)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=5)   # mirror conv2
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)    # mirror conv1

    # -------- Encoder --------
    def extract_adapter_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                  # [B,10,12,12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # [B,20,4,4]
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        adapter_f = F.relu(self.adapter(x))                         # [B, f_size]
        return adapter_f

    def forward_from_adapter(self, adapter_f):
        z = F.relu(self.fc2(adapter_f))                             # [B, 16]
        logits = self.fc3(F.normalize(z, p=2, dim=1))               # [B, num_classes] classification
        return logits

    # -------- Decoder --------
    def decode(self, f):
        x = F.relu(self.fc_dec1(f))
        x = F.relu(self.fc_dec2(x))
        x = x.view(-1, 20, 4, 4)
        x = F.relu(self.deconv1(F.interpolate(x, scale_factor=2))) # [B,10,10,10] approx
        x = torch.sigmoid(self.deconv2(F.interpolate(x, scale_factor=2))) # [B,1,28,28]
        return x

    # -------- Full forward --------
    def forward(self, x):
        adapter_f = self.extract_adapter_features(x)
        logits = self.forward_from_adapter(adapter_f)
        recon = self.decode(adapter_f)
        return logits, recon, adapter_f

