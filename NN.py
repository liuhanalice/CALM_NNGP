import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import random
import os
import matplotlib.pyplot as plt

import NN_utils

# ---- Global Config ----
GP_doubleSoftmax = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# num_tasks = 6
# global_task_labels = ["Task 0: 0-4", "Task 1: 5", "Task 2: 6", "Task 3: 7", "Task 4: 8", "Task 5: 9"]
# old_logits_dict = {}
# tasks_epochs_accuracy = [[] for _ in range(num_tasks)] # list of lists (rows: tasks, cols: epochs)
# tasks_epochs_accuracy_train = [[] for _ in range(num_tasks)]

# seed = 42
# root_folder = "500_Indpoints_withGP_f64(generate500-GP-use-Nonfiltered)"
# f_size = 64
# # each task training config
# train_batch_size = 128
# test_batch_size = 128
# epochs = 30


# ---- Model Architecture ----
class CALM_NN(nn.Module):
    def __init__(self, f_size = 64, num_classes = 10):
        super(CALM_NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 160)
        self.adapter = nn.Linear(160, f_size)
        self.fc2 = nn.Linear(f_size, 16)
        self.fc3 = nn.Linear(16, num_classes)
        
    def extract_adapter_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        adapter_f = F.relu(self.adapter(x))
        return adapter_f
    
    def forward_from_adapter(self, adapter_f):
        x = F.relu(self.fc2(adapter_f))
        logits = self.fc3(F.normalize(x, p=2, dim=1))
        return logits   
    
    def forward(self, x):
        adapter_f = self.extract_adapter_features(self, x)
        logits = self.forward_from_adapter(self, adapter_f)
        return logits



# ---- Utils Functions ----
def custom_loss_fn(logits, targets, old_logits_dict=None, inputs=None, alpha=1.0):
    ce_loss = F.cross_entropy(logits, targets)

    if old_logits_dict is None or inputs is None:
        return ce_loss
    
    logit_reg_loss = 0
    count = 0
    for i, x in enumerate(inputs):
        x_id = x.cpu().numpy.tobytes()
        if x_id in old_logits_dict:
            prev_logits = old_logits_dict[x_id].to(logits.device)
            logit_reg_loss += F.mse_loss(logits[i], prev_logits)
            count += 1
    
    if count > 0:
        logit_reg_loss = logit_reg_loss / count
        return ce_loss + alpha * logit_reg_loss
    
    return ce_loss


def evaluate_clf_accuracy(model, dataloader, device=device):
    model.eval()
    correct, total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_logit_preservation(model, dataloader, old_logits_dict, device="cuda"):
    model.eval()
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            logits = model(data)
            for i, x in enumerate(data):
                x_id = x.detach().cpu().numpy.tobytes()
                if x_id in old_logits_dict:
                    prev_logits = old_logits_dict[x_id].to(logits.device)
                    total_mse += F.mse_loss(logits[i], prev_logits).item()
                    count += 1
    return total_mse / count if count > 0 else None


def evaluate_tasks(model, tsloaders, task_id, num_tasks, device="cuda"):
    """
    Returns a list of length num_tasks with classification accuracy for tasks <= task_id
    and np.nan for unseen tasks > task_id.
    """
    accs = []
    for task in range(num_tasks):
        if task <= task_id:
            acc = evaluate_clf_accuracy(model, tsloaders[task], device=device)
            accs.append(acc)
        else:
            accs.append(np.nan)
    return accs


def train(model, trloader, epochs, learning_rate=1e-3, alpha=1.0, old_logits_dict=None, device="cuda"):
    """
    
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    # criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (data, target) in enumerate(trloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            
            loss = custom_loss_fn(
                logits=logits,
                targets=target,
                old_logits_dict=old_logits_dict,
                inputs=data,
                alpha=alpha
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(trloader)
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        
    return history



def test(model, tsloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in tsloader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / len(tsloader)
    print(f"Test: Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return acc

def test(model, dataloader, device="cuda", old_logits_dict=None):
    """
    Evaluate a trained model on a dataset.
    
    Parameters:
        model: trained nn.Module
        dataloader: torch.utils.data.DataLoader
        device: device to run on
        old_logits_dict: optional dict {x_bytes -> logits} to measure preservation
    
    Returns:
        results: dict with 'loss', 'accuracy', and optionally 'logit_mse'
    """
    model.eval()
    model.to(device)
    total_loss, correct, total = 0.0, 0, 0
    logit_mse, count = 0.0, 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            
            # classification loss (pure CE for evaluation)
            loss = F.cross_entropy(logits, target, reduction="sum")
            total_loss += loss.item()

            # accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            # optional: logit preservation
            if old_logits_dict is not None:
                for i, x in enumerate(data):
                    x_id = x.detach().cpu().numpy().tobytes()
                    if x_id in old_logits_dict:
                        prev_logits = old_logits_dict[x_id].to(device)
                        logit_mse += F.mse_loss(logits[i], prev_logits).item()
                        count += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    results = {"loss": avg_loss, "accuracy": acc}

    if old_logits_dict is not None and count > 0:
        results["logit_mse"] = logit_mse / count

    return results


def extract_adapter_features_and_labels(model, data_loader, device='cpu'):
    model.eval()
    features = []
    scores = []
    labels = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            adapter_f = model.extract_adapter_features(data)
            out = model.forward_from_adapter(adapter_f)

            features.append(adapter_f.cpu())
            scores.append(out.cpu())
            labels.append(target.cpu())
    
    features = torch.cat(features, dim=0)
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, scores, labels


def train_from_adapter_features(model, trloader, task_id, tsloaders, epochs=10, learning_rate=0.001):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze fc2 and fc3 for training
    for param in model.fc2.parameters():
        param.requires_grad = True
    for param in model.fc3.parameters():
        param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (adapter_f, target) in enumerate(trloader):
            adapter_f, target = adapter_f.to(device), target.to(device)  # adapter_f is input

            optimizer.zero_grad()
            output = model.forward_from_adapter(adapter_f)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        acc = 100. * correct / total
        avg_loss = total_loss / len(trloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

        # Evaluate per-task
        for task in range(num_tasks):
            if task <= task_id:
                acc_test = test(model, tsloaders[task], device=device)
                tasks_epochs_accuracy[task].append(acc_test)
                tasks_epochs_accuracy_train[task].append(acc)
            else:
                tasks_epochs_accuracy[task].append(np.nan)
                tasks_epochs_accuracy_train[task].append(np.nan)


def extract_from_adapter_features(model, dataloader, device='cpu'):
    """
    Given a dataloader that yields (adapter_features, labels),
    compute model scores and return (features, scores, labels).
    """
    model.eval()
    all_features = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for adapter_f, target in dataloader:
            adapter_f, target = adapter_f.to(device), target.to(device)

            output = model.forward_from_adapter(adapter_f)  # directly from adapter input

            all_features.append(adapter_f.cpu())  
            all_scores.append(output.cpu())       
            all_labels.append(target.cpu())

    features = torch.cat(all_features, dim=0)
    logits = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, logits, labels




