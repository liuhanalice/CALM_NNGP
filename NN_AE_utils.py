import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import random
import os
import matplotlib.pyplot as plt

from NN_AE import CALM_AE_NN

# -----------------------------
# Helper Functions
# -----------------------------
def _make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)

def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_classifier_head(model: CALM_AE_NN):
    """Freeze fc2 and fc3 so f_size→logits mapping preserved for old classes"""
    _set_requires_grad(model.fc2, False)
    _set_requires_grad(model.fc3, False)

def unfreeze_classifier_head(model: CALM_AE_NN):
    _set_requires_grad(model.fc2, True)
    _set_requires_grad(model.fc3, True)

def freeze_encoder_decoder(model: CALM_AE_NN):
    # encoder
    _set_requires_grad(model.conv1, False)
    _set_requires_grad(model.conv2, False)
    _set_requires_grad(model.fc1,  False)
    _set_requires_grad(model.adapter, False)
    # decoder
    _set_requires_grad(model.fc_dec1, False)
    _set_requires_grad(model.fc_dec2, False)
    _set_requires_grad(model.deconv1, False)
    _set_requires_grad(model.deconv2, False)

def unfreeze_encoder_decoder(model: CALM_AE_NN):
    # encoder
    _set_requires_grad(model.conv1, True)
    _set_requires_grad(model.conv2, True)
    _set_requires_grad(model.fc1,  True)
    _set_requires_grad(model.adapter, True)
    # decoder
    _set_requires_grad(model.fc_dec1, True)
    _set_requires_grad(model.fc_dec2, True)
    _set_requires_grad(model.deconv1, True)
    _set_requires_grad(model.deconv2, True)

# ---------- Build feature dict BEFORE training the new task ----------
@torch.no_grad()
def build_feature_dict(model, dataloader, device=None, max_items=None, dtype=torch.float32):
    """
    Capture snapshots:
      - feat_dict: input image bytes -> feature vector f
      - logit_dict: input image bytes -> logits (pre-softmax)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    feat_dict, logit_dict = {}, {}
    seen = 0
    for data, _ in dataloader:
        data = data.to(device)
        # forward returns: logits, recon, f
        logits, _, f = model(data)
        f = f.detach().to("cpu", dtype=dtype)
        logits = logits.detach().to("cpu", dtype=dtype)

        for i in range(data.size(0)):
            key = data[i].detach().cpu().numpy().tobytes()
            feat_dict[key] = f[i].clone()
            logit_dict[key] = logits[i].clone()
            seen += 1
            if max_items is not None and seen >= max_items:
                return feat_dict, logit_dict
    return feat_dict, logit_dict

# -----------------------------
# Train / Test 
# -----------------------------
# -------- Train (CE + optional reconstruction loss from f_size) --------

# NOTE: This version is no longer used. Use train_2_stage instead.
def train( model,
    trloader,
    epochs=5,
    lr=1e-3,
    lambda_rec=0.0,       # reconstruction loss weight
    lambda_feat=1.0,      # feature preservation weight (MSE between current f and cached f_prev)
    lambda_logit=1.0,     # logit preservation weight (MSE between current ;ogits and cached logits)
    old_feature_dict=None,
    old_logit_dict=None,
    logit_mask=None,      # 1/0 mask for classes to preserve (size [num_classes]); None = all
    freeze_head=False,
    device=None,
    grad_clip=None,        # e.g., 1.0 or None,
    eval_fn=None           # when provided, return dict include current test accuracy based on eval_fn provided
):
    """
    Loss = CE(y, logits) + λ_rec * MSE(recon, x) + λ_feat * MSE(f_curr, f_prev) {over batch matches} + λ_logit* MSE(z_curr, z_prev) {over batch matches}.
    Only adds the feature and logit term for samples found in the old_feature_dict/old_logit_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if freeze_head:
        freeze_classifier_head(model)
    else: # for task 0 [update already, all task unfreeze]
        unfreeze_classifier_head(model)

    optimizer = _make_optimizer(model, lr)

    history = {"loss": [], "ce": [], "rec": [], "feat_reg": [], "logit_reg": [], "acc": []}

    if logit_mask is not None:
        logit_mask = logit_mask.to(device).bool()

    # Epochs
    for ep in range(epochs):
        model.train()
        total_loss = total_ce = total_rec = total_feat = total_logit = 0.0
        correct = total = 0
        test_acc = None

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
            
            # --- logit preservation term ---
            logit_reg = torch.tensor(0.0, device=device)
            if old_logit_dict is not None and lambda_logit > 0:
                reg_sum = torch.tensor(0.0, device=device)
                match_count = 0
                with torch.no_grad():
                    keys = [data[i].detach().cpu().numpy().tobytes() for i in range(data.size(0))]
                for i, k in enumerate(keys):
                    if k in old_logit_dict:
                        z_prev = old_logit_dict[k].to(device)  # [num_classes], should be *logits* from old model
                        if logit_mask is not None:
                            z_curr_masked = logits[i][logit_mask]
                            z_prev_masked = z_prev[logit_mask]
                            reg_sum = reg_sum + F.mse_loss(z_curr_masked, z_prev_masked)
                        else:
                            reg_sum = reg_sum + F.mse_loss(logits[i], z_prev)
                        match_count += 1
                if match_count > 0:
                    logit_reg = reg_sum / match_count

            loss = ce + lambda_rec * rec + lambda_feat * feat_reg + lambda_logit * logit_reg
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_rec  += (rec.item() if lambda_rec > 0 else 0.0)
            total_feat += (feat_reg.item() if (old_feature_dict is not None and lambda_feat > 0) else 0.0)
            total_logit+= (logit_reg.item() if (old_logit_dict is not None and lambda_logit > 0) else 0.0)

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total   += label.size(0)

        # eval per epoch
        if eval_fn is not None:
            test_acc = eval_fn(model)  # expects 0..1
            history.setdefault("test_acc", []).append(100.0 * test_acc)

        n_batches = max(len(trloader), 1)
        epoch_loss = total_loss / n_batches
        epoch_ce   = total_ce   / n_batches
        epoch_rec  = total_rec  / n_batches if lambda_rec > 0 else 0.0
        epoch_feat = total_feat / n_batches if (old_feature_dict is not None and lambda_feat > 0) else 0.0
        epoch_logit = total_logit/ n_batches if (old_logit_dict  is not None and lambda_logit > 0) else 0.0
        epoch_acc  = 100.0 * correct / max(total, 1)

        history["loss"].append(epoch_loss)
        history["ce"].append(epoch_ce)
        history["rec"].append(epoch_rec)
        history["feat_reg"].append(epoch_feat)
        history["logit_reg"].append(epoch_logit)
        history["acc"].append(epoch_acc)

        msg = (
            f"Epoch {ep+1}/{epochs} | loss {epoch_loss:.4f} | ce {epoch_ce:.4f} "
            f"| rec {epoch_rec:.4f} | feat {epoch_feat:.4f} | logit {epoch_logit:.4f} "
            f"| train_acc {epoch_acc:.2f}%"
        )
        msg += f" | test_acc {test_acc*100:.2f}%" if test_acc is not None else " | test_acc N/A"
        print(msg)

    return history


def train_2_stage(
    model: CALM_AE_NN,
    trloader,
    # --- Stage 1 (AE) ---
    epochs_stage1=3,
    lr_stage1=1e-3,
    lambda_rec_stage1=1.0,     # recon loss weight
    lambda_feat_stage1=1.0,    # feature preservation weight (MSE(f_curr, f_prev))
    old_feature_dict=None,     # {image_bytes: f_prev_tensor}
    # --- Stage 2 (Head) ---
    epochs_stage2=3,
    lr_stage2=1e-3,
    lambda_ce_stage2=1.0,      # CE weight
    lambda_logit_stage2=1.0,   # logit preservation weight (MSE(z_curr, z_prev[mask]))
    old_logit_dict=None,       # {image_bytes: z_prev_tensor} (prev logits)
    logit_mask=None,           # torch.BoolTensor[num_classes] or None
    # --- common ---
    device=None,
    grad_clip=None,
    eval_fn=None               # callable(model)->retuen *list* of per-task accs in [0,1]
):
    """
    Stage 1 (AE): freeze head; train En+De with loss:
        L1 = λ_rec * MSE(recon, x) + λ_feat * MSE(f_curr, f_prev) on matched samples.
    Stage 2 (Head): freeze En+De; train head with:
        L2 = λ_ce * CE(y, logits) + λ_logit * MSE(z_curr, z_prev[mask]) on matched samples.
    Returns history dict with per-epoch metrics for both stages.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if logit_mask is not None:
        logit_mask = logit_mask.to(device).bool()

    history = {
        "stage": [], "epoch": [], "loss": [],
        "ce": [], "rec": [], "feat_reg": [], "logit_reg": [],
        "train_acc": [], "test_acc": [],
    }

    # =========================
    # Stage 1: AE training
    # =========================
    freeze_classifier_head(model)
    unfreeze_encoder_decoder(model)
    opt1 = _make_optimizer(model, lr_stage1)

    for ep in range(epochs_stage1):
        model.train()
        total_loss = total_rec = total_feat = 0.0
        correct = total = 0
        test_acc = None

        for data, label in trloader:
            data, label = data.to(device), label.to(device)

            opt1.zero_grad()
            logits, recon, f = model(data)  # head is frozen; used only for monitoring acc

            # recon loss
            rec = F.mse_loss(recon, data) if lambda_rec_stage1 > 0 else torch.tensor(0.0, device=device)

            # feature preservation (only for keys we have)
            feat_reg = torch.tensor(0.0, device=device)
            if old_feature_dict is not None and lambda_feat_stage1 > 0:
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

            loss = lambda_rec_stage1 * rec + lambda_feat_stage1 * feat_reg
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt1.step()

            total_loss += loss.item()
            total_rec  += (rec.item() if lambda_rec_stage1 > 0 else 0.0)
            total_feat += (feat_reg.item() if (old_feature_dict is not None and lambda_feat_stage1 > 0) else 0.0)

            # monitor acc (head is frozen; just for visibility)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total   += label.size(0)

        test_accs_seen = None
        test_acc_mean = None
        if eval_fn is not None:
            test_accs_seen = eval_fn(model) # list of per-task accs
            est_acc_mean  = float(np.mean(test_accs_seen)) if len(test_accs_seen) else None

        n_batches = max(len(trloader), 1)
        epoch_loss = total_loss / n_batches
        epoch_rec  = total_rec  / n_batches if lambda_rec_stage1 > 0 else 0.0
        epoch_feat = total_feat / n_batches if (old_feature_dict is not None and lambda_feat_stage1 > 0) else 0.0
        epoch_acc  = 100.0 * correct / max(total, 1)

        history["stage"].append("AE") # stage 1: AE
        history["epoch"].append(ep + 1) # gliobal epoch count
        history["loss"].append(epoch_loss) # total loss (AE recon loss + feat preserve loss)
        history["ce"].append(0.0) # classification CE loss (not for stage 1)
        history["rec"].append(epoch_rec) # AE recon loss
        history["feat_reg"].append(epoch_feat) # feature preserve
        history["logit_reg"].append(0.0) # logit preserve (not for stage 1)
        history["train_acc"].append(epoch_acc) # train acc (head is frozen; just for visibility)
        # history["test_acc_newtask"].append(100.0 * test_acc if test_acc is not None else None) # this is mean, = test_acc_mean
        history.setdefault("test_accs_seen", []).append(test_accs_seen)   # list or None
        history.setdefault("test_acc_mean", []).append(test_acc_mean)    # scalar or None

        msg = (f"[Stage1/AE] Epoch {ep+1}/{epochs_stage1} | loss {epoch_loss:.4f} "
               f"| rec {epoch_rec:.4f} | feat {epoch_feat:.4f} | train_acc {epoch_acc:.2f}%")
        msg += f" | test_acc {100.0*test_acc:.2f}%" if test_acc is not None else ""
        print(msg)

    # =========================
    # Stage 2: Head training
    # =========================
    freeze_encoder_decoder(model)
    unfreeze_classifier_head(model)
    opt2 = _make_optimizer(model, lr_stage2)

    for ep in range(epochs_stage2):
        model.train()
        total_loss = total_ce = total_logit = 0.0
        correct = total = 0
        test_acc = None

        for data, label in trloader:
            data, label = data.to(device), label.to(device)

            opt2.zero_grad()
            logits, recon, f = model(data)  # En is frozen; recon unused here

            # CE on all samples
            ce = F.cross_entropy(logits, label) * lambda_ce_stage2

            # logit preservation for matched samples (masked if provided)
            logit_reg = torch.tensor(0.0, device=device)
            if old_logit_dict is not None and lambda_logit_stage2 > 0:
                reg_sum = torch.tensor(0.0, device=device)
                match_count = 0
                with torch.no_grad():
                    keys = [data[i].detach().cpu().numpy().tobytes() for i in range(data.size(0))]
                for i, k in enumerate(keys):
                    if k in old_logit_dict:
                        z_prev = old_logit_dict[k].to(device)  # [num_classes], logits
                        if logit_mask is not None:
                            z_curr_masked = logits[i][logit_mask]
                            z_prev_masked = z_prev[logit_mask]
                            reg_sum = reg_sum + F.mse_loss(z_curr_masked, z_prev_masked)
                        else:
                            reg_sum = reg_sum + F.mse_loss(logits[i], z_prev)
                        match_count += 1
                if match_count > 0:
                    logit_reg = reg_sum / match_count
                logit_reg = lambda_logit_stage2 * logit_reg

            loss = ce + logit_reg
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt2.step()

            total_loss += loss.item()
            total_ce   += ce.item()
            total_logit+= (logit_reg.item() if (old_logit_dict is not None and lambda_logit_stage2 > 0) else 0.0)

            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total   += label.size(0)

        test_accs_seen = None
        test_acc_mean = None
        if eval_fn is not None:
            test_accs_seen = eval_fn(model) # list of per-task accs
            est_acc_mean  = float(np.mean(test_accs_seen)) if len(test_accs_seen) else None

        n_batches = max(len(trloader), 1)
        epoch_loss  = total_loss / n_batches
        epoch_ce    = total_ce   / n_batches
        epoch_logit = total_logit/ n_batches if (old_logit_dict is not None and lambda_logit_stage2 > 0) else 0.0
        epoch_acc   = 100.0 * correct / max(total, 1)

        history["stage"].append("Head")
        history["epoch"].append(ep + 1)
        history["loss"].append(epoch_loss)
        history["ce"].append(epoch_ce)
        history["rec"].append(0.0)
        history["feat_reg"].append(0.0)
        history["logit_reg"].append(epoch_logit) # preserve old_logit
        history["train_acc"].append(epoch_acc)
        history["test_acc"].append(100.0 * test_acc if test_acc is not None else None)
        history.setdefault("test_accs_seen", []).append(test_accs_seen)   # list or None
        history.setdefault("test_acc_mean", []).append(test_acc_mean)    # scalar or None

        msg = (f"[Stage2/Head] Epoch {ep+1}/{epochs_stage2} | loss {epoch_loss:.4f} "
               f"| ce {epoch_ce:.4f} | logit {epoch_logit:.4f} | train_acc {epoch_acc:.2f}%")
        msg += f" | test_acc {100.0*test_acc:.2f}%" if test_acc is not None else ""
        print(msg)

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