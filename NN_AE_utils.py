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
# Helper Functions
# -----------------------------
def freeze_classifier_head(model):
    """Freeze fc2 and fc3 so f_size→logits mapping stays fixed."""
    for p in model.fc2.parameters():
        p.requires_grad = False
    for p in model.fc3.parameters():
        p.requires_grad = False

def unfreeze_classifier_head(model):
    for p in model.fc2.parameters():
        p.requires_grad = True
    for p in model.fc3.parameters():
        p.requires_grad = True

def _make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)

# ---------- Build feature dict BEFORE training the new task ----------
@torch.no_grad()
def build_feature_dict(model, dataloader, device=None, max_items=None, dtype=torch.float32):
    """
    Capture a snapshot mapping from input image bytes -> previous f_size features.
    Use model.eval() to disable dropout.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    feat_dict = {}
    seen = 0
    for data, _ in dataloader:
        data = data.to(device)
        # forward returns: logits, recon, f
        _, _, f = model(data)
        f = f.detach().to("cpu", dtype=dtype)

        for i in range(data.size(0)):
            key = data[i].detach().cpu().numpy().tobytes()
            feat_dict[key] = f[i].clone()
            seen += 1
            if max_items is not None and seen >= max_items:
                return feat_dict
    return feat_dict

# -----------------------------
# Train / Test 
# -----------------------------
# -------- Train (CE + optional reconstruction loss from f_size) --------
def train( model,
    trloader,
    epochs=5,
    lr=1e-3,
    lambda_rec=0.0,       # reconstruction loss weight
    lambda_feat=1.0,      # feature preservation weight (MSE between current f and cached f_prev)
    old_feature_dict=None,
    freeze_head=False,
    device=None,
    grad_clip=None        # e.g., 1.0 or None
):
    """
    Loss = CE(y, logits) + λ_rec * MSE(recon, x) + λ_feat * MSE(f_curr, f_prev) over batch matches.
    Only adds the feature term for samples found in the old_feature_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if freeze_head:
        freeze_classifier_head(model)
    else: # for task 0
        unfreeze_classifier_head(model)

    optimizer = _make_optimizer(model, lr)

    history = {"loss": [], "ce": [], "rec": [], "feat_reg": [], "acc": []}

    for ep in range(epochs):
        model.train()
        total_loss = total_ce = total_rec = total_feat = 0.0
        correct = total = 0

        for data, label in trloader:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            logits, recon, f = model(data)

            ce  = F.cross_entropy(logits, label)
            rec = F.mse_loss(recon, data) if lambda_rec > 0 else torch.tensor(0.0, device=device)

            # --- feature preservation term ---
            feat_reg = torch.tensor(0.0, device=device)
            if old_feature_dict is not None and lambda_feat > 0:
                # accumulate MSE over matches in the batch
                reg_sum = torch.tensor(0.0, device=device)
                match_count = 0
                with torch.no_grad():
                    keys = [data[i].detach().cpu().numpy().tobytes() for i in range(data.size(0))]
                for i, k in enumerate(keys):
                    if k in old_feature_dict:
                        f_prev = old_feature_dict[k].to(device)  # [f_size]
                        reg_sum = reg_sum + F.mse_loss(f[i], f_prev)
                        match_count += 1
                if match_count > 0:
                    feat_reg = reg_sum / match_count

            loss = ce + lambda_rec * rec + lambda_feat * feat_reg
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_rec  += (rec.item() if lambda_rec > 0 else 0.0)
            total_feat += (feat_reg.item() if (old_feature_dict is not None and lambda_feat > 0) else 0.0)

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total   += label.size(0)

        n_batches = max(len(trloader), 1)
        epoch_loss = total_loss / n_batches
        epoch_ce   = total_ce   / n_batches
        epoch_rec  = total_rec  / n_batches if lambda_rec > 0 else 0.0
        epoch_feat = total_feat / n_batches if (old_feature_dict is not None and lambda_feat > 0) else 0.0
        epoch_acc  = 100.0 * correct / max(total, 1)

        history["loss"].append(epoch_loss)
        history["ce"].append(epoch_ce)
        history["rec"].append(epoch_rec)
        history["feat_reg"].append(epoch_feat)
        history["acc"].append(epoch_acc)

        print(f"Epoch {ep+1}/{epochs} | loss {epoch_loss:.4f} | ce {epoch_ce:.4f} "
              f"| rec {epoch_rec:.4f} | feat {epoch_feat:.4f} | acc {epoch_acc:.2f}%")

    return history

# -------- Test (plain CE; optional recon loss reporting) --------
@torch.no_grad()
def test(model, dataloader, device=None, report_recon=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval(); model.to(device)

    total_ce, total, correct = 0.0, 0, 0
    total_rec = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, recon, f = model(x)

        ce = F.cross_entropy(logits, y, reduction="sum")
        total_ce += ce.item()

        if report_recon:
            total_rec += F.mse_loss(recon, x, reduction="sum").item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

    avg_ce = total_ce / max(total, 1)
    acc    = correct / max(total, 1)
    out = {"ce_loss": avg_ce, "accuracy": acc}
    if report_recon:
        out["rec_loss"] = total_rec / max(total, 1)
    return out