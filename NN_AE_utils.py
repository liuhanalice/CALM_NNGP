import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
import random
import os
import copy
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

# NOTE: This version is no longer used. 
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


# NOTE: This version is no longer used. 
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
            test_accs_seen = [acc * 100 for acc in test_accs_seen]
            test_acc_mean  = float(np.mean(test_accs_seen)) if len(test_accs_seen) else None

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
            test_accs_seen = [acc * 100 for acc in test_accs_seen]
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


# The logit extension map
def phi_extend(z_prev, Kt, fill="mean"):
    """
    Extend logit vector to have Kt size

    fill method:
    1) mean  2) zer0  3) const:{c}
    """
    # z_prev: [B, K_prev]
    B, K_prev = z_prev.shape
    if Kt == K_prev:
        return z_prev
    if fill == "mean":
        # c = z_prev.mean(dim=1, keepdim=True)
        if K_prev <= 1:
            c = torch.zeros(B, 1, device=z_prev.device, dtype=z_prev.dtype)
        else:
            max_idx = z_prev.argmax(dim=1, keepdim=True)  # [B,1]
            # mask out the max column
            mask = torch.ones_like(z_prev, dtype=torch.bool)
            mask.scatter_(1, max_idx, False)              # False at max positions
            # sum remaining logits and divide by (K_prev - 1)
            sum_excl = (z_prev * mask.to(z_prev.dtype)).sum(dim=1, keepdim=True)
            c = sum_excl / (K_prev - 1)
    elif fill == "zero":
        c = torch.zeros(B, 1, device=z_prev.device, dtype=z_prev.dtype)
    elif fill.startswith("const:"):
        val = float(fill.split(":", 1)[1])
        c = torch.full((B,1), val, device=z_prev.device, dtype=z_prev.dtype)
    else:
        raise ValueError(fill)
    pad = c.repeat(1, Kt - K_prev)
    return torch.cat([z_prev, pad], dim=1)



def train_2_stage_class_aware(
    model,
    teacher_model,          # frozen snapshot from previous task
    trloader,
    old_classes,            # iterable of ints (classes seen before task t)
    new_classes,            # iterable of ints (classes introduced at task t)
    # stage 1
    epochs_stage1=3,
    lr_stage1=1e-3,
    lambda_rec=1.0,
    lambda_feat=1.0,
    rec_on="new",           # "new" or "all"
    # stage 2
    epochs_stage2=3,
    lr_stage2=1e-3,
    lambda_ce=1.0,
    lambda_logit=1.0,
    ce_on="new",            # "new" or "all"
    phi_fill="mean",        # optional mask over logits (explicitly choose which columns of logits to preserve)
    # common
    device=None,
    grad_clip=None,
    eval_fn=None,
):
    """
    Two-Stage Class-Incremental Training

    This function implements the following training strategy:

    ------------------------------------------------------------
    Stage 1: Autoencoder Adaptation (Encoder + Decoder updated)
    ------------------------------------------------------------
        Goal:
            Adapt representation to new-task data
            while preserving feature space for OLD classes.

        Head is frozen.

        Loss:
            L_stage1 =
                λ_rec  * Reconstruction Loss (on NEW or ALL samples)
              + λ_feat * Feature Distillation Loss (OLD samples only)

        Feature distillation:
            MSE(f_current(x_old), f_teacher(x_old))

    ------------------------------------------------------------
    Stage 2: Classifier Head Adaptation
    ------------------------------------------------------------
        Goal:
            Learn to classify new classes
            while preserving decision boundary for old classes.

        Encoder/Decoder are frozen.

        Loss:
            L_stage2 =
                λ_ce     * CrossEntropy (NEW or ALL samples)
              + λ_logit  * Logit Distillation (OLD samples only)

        Logit distillation:
            MSE(z_current(x_old), z_teacher(x_old))
            (optionally masked to OLD classes only)

    ------------------------------------------------------------
    OLD vs NEW class split is determined using true labels.
    No per-sample dictionary matching is used.
    Teacher model provides preservation targets dynamically.

    Returns:
        history dictionary containing per-epoch metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    teacher_model.to(device).eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    old_classes = list(map(int, old_classes))
    new_classes = list(map(int, new_classes))

    old_set = set(old_classes)
    new_set = set(new_classes)

    old_cols = sorted(old_classes)                       # columns in teacher/student corresponding to old classes
    seen_cols = sorted(list(old_set.union(new_set)))     # columns corresponding to all seen classes (old + current new)
    K_prev = len(old_cols)
    K_seen = len(seen_cols)

    # map global class id -> position in old_cols
    old_pos = {c: i for i, c in enumerate(old_cols)}

    history = {
        "stage": [],
        "epoch": [],
        "loss": [],
        "rec": [],
        "feat_reg": [],
        "ce": [],
        "logit_reg": [],
        "train_acc": [],            
        "test_accs_seen": [],       # list-of-lists per epoch (if eval_fn provided)
        "test_acc_mean": [],        # scalar mean (if eval_fn provided)
    }

    # ----------------
    # Stage 1: AE
    # ----------------
    freeze_classifier_head(model)
    unfreeze_encoder_decoder(model)
    opt1 = _make_optimizer(model, lr_stage1)

    for ep in range(epochs_stage1):
        model.train()
        total_loss = total_rec = total_feat = 0.0
        correct = total = 0
        n_batches = 0

        for x, y in trloader:
            x, y = x.to(device), y.to(device)
            # Determine samples belong to old or new classes
            y_list = y.detach().cpu().tolist()
            new_mask = torch.tensor([yy in new_set for yy in y_list], device=device, dtype=torch.bool)
            old_mask = torch.tensor([yy in old_set for yy in y_list], device=device, dtype=torch.bool)

            opt1.zero_grad()
            logits, recon, feat = model(x)

            # ---------------- Reconstruction Loss ----------------
            rec = torch.tensor(0.0, device=device)

            if lambda_rec > 0:
                if rec_on == "all":
                    rec = F.mse_loss(recon, x)
                elif rec_on == "new" and new_mask.any():
                    rec = F.mse_loss(recon[new_mask], x[new_mask])


            # ---------------- Feature Preservation ----------------
            feat_reg = torch.tensor(0.0, device=device)

            if lambda_feat > 0 and old_mask.any():
                with torch.no_grad():
                    _, _, feat_teacher = teacher_model(x[old_mask])
                feat_reg = F.mse_loss(feat[old_mask], feat_teacher)

            # Combined loss
            loss = lambda_rec * rec + lambda_feat * feat_reg
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt1.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_feat += feat_reg.item()
            n_batches += 1

            # train acc (full head)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # Evaluation
        test_accs_seen = None
        test_acc_mean = None
        if eval_fn is not None:
            test_accs_seen = eval_fn(model)  # list of per-task accs
            test_accs_seen = [acc * 100 for acc in test_accs_seen]
            test_acc_mean = float(np.mean(test_accs_seen)) if len(test_accs_seen) else None

        history["stage"].append("AE")
        history["epoch"].append(ep + 1)
        history["loss"].append(total_loss / max(n_batches, 1))
        history["rec"].append(total_rec / max(n_batches, 1))
        history["feat_reg"].append(total_feat / max(n_batches, 1))
        history["ce"].append(0.0) # this is for stage 2
        history["logit_reg"].append(0.0) # this is for stage 2
        history["train_acc"].append(100.0 * correct / max(total, 1))
        history["test_accs_seen"].append(test_accs_seen)
        history["test_acc_mean"].append(test_acc_mean)

        print(f"[Stage1/AE] ep {ep+1}/{epochs_stage1} | loss {history['loss'][-1]:.4f} "
              f"| rec {history['rec'][-1]:.4f} | feat {history['feat_reg'][-1]:.4f} "
              f"| train_acc {history['train_acc'][-1]:.2f}%")

    # ----------------
    # Stage 2: Head
    # ----------------
    freeze_encoder_decoder(model)
    unfreeze_classifier_head(model)
    opt2 = _make_optimizer(model, lr_stage2)

    for ep in range(epochs_stage2):
        model.train()
        total_loss = total_ce = total_logit = 0.0
        correct = total = 0
        n_batches = 0

        for x, y in trloader:
            x, y = x.to(device), y.to(device)
            y_list = y.detach().cpu().tolist()
            new_mask = torch.tensor([yy in new_set for yy in y_list], device=device, dtype=torch.bool)
            old_mask = torch.tensor([yy in old_set for yy in y_list], device=device, dtype=torch.bool)

            opt2.zero_grad()
            logits, recon, feat = model(x)

            # ---------------- Cross-Entropy ----------------
            ce = torch.tensor(0.0, device=device)

            if lambda_ce > 0:
                if ce_on == "all":
                    ce = F.cross_entropy(logits, y)
                elif ce_on == "new" and new_mask.any():
                    ce = F.cross_entropy(logits[new_mask], y[new_mask])

            # ---------------- Logit Preservation ----------------
            logit_reg = torch.tensor(0.0, device=device)

            if lambda_logit > 0 and old_mask.any() and K_prev > 0:
                x_old = x[old_mask]
                y_old = y[old_mask]

                with torch.no_grad():
                    z_teacher_full, _, _ = teacher_model(x_old)
                
                z_teacher_old = z_teacher_full[:, old_cols]
                z_teacher_phi = phi_extend(z_teacher_old, Kt=K_seen, fill=phi_fill)

                # student logits restricted to SEEN columns (old+new)
                z_student_seen = logits[old_mask][:, seen_cols]  # [B_old, K_seen]
                logit_reg = F.mse_loss(z_student_seen, z_teacher_phi)
                # logit_reg = F.mse_loss( # apply on softmax (logits)
                #     F.softmax(z_student_seen, dim=1),
                #     F.softmax(z_teacher_phi, dim=1)
                # )


            loss = lambda_ce * ce + lambda_logit * logit_reg
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt2.step()

            total_loss += loss.item()
            total_ce += ce.item()
            total_logit += logit_reg.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # Evaluation
        test_accs_seen = None
        test_acc_mean = None
        if eval_fn is not None:
            test_accs_seen = eval_fn(model)
            test_accs_seen = [acc * 100 for acc in test_accs_seen]
            test_acc_mean = float(np.mean(test_accs_seen)) if len(test_accs_seen) else None

        history["stage"].append("Head")
        history["epoch"].append(ep + 1)
        history["loss"].append(total_loss / max(n_batches, 1))
        history["rec"].append(0.0) # this is only for stage1
        history["feat_reg"].append(0.0) # this is only for stage1
        history["ce"].append(total_ce / max(n_batches, 1))
        history["logit_reg"].append(total_logit / max(n_batches, 1))
        history["train_acc"].append(100.0 * correct / max(total, 1))
        history["test_accs_seen"].append(test_accs_seen)
        history["test_acc_mean"].append(test_acc_mean)

        print(f"[Stage2/Head] ep {ep+1}/{epochs_stage2} | loss {history['loss'][-1]:.4f} "
              f"| ce {history['ce'][-1]:.4f} | logit {history['logit_reg'][-1]:.4f} "
              f"| train_acc {history['train_acc'][-1]:.2f}%")

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