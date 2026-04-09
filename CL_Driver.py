# run_continual_mnist_driver.py
from asyncio import tasks
from asyncio import tasks
import os
import csv
import json
import time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import re
import matplotlib.pyplot as plt
import shlex
import shutil
import subprocess
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset, Dataset
from torchvision import datasets, transforms



from NN_AE import CALM_AE_NN
from NN_AE_utils import (
    train,  # CE + λ_rec*MSE(recon,x) + λ_feat*MSE(f,f_prev)
    test,                             # eval CE (and optional recon report)
    build_feature_dict,               # caches {image_bytes -> f_size feature}
    train_2_stage,
    train_2_stage_class_aware
)

# =============== Costomized Dataset Class: Matching with MNIST format ===============
class RecoveredDataset(Dataset):
    def __init__(self, images, labels):
        # images: torch.FloatTensor [N,1,28,28]
        # labels: torch.LongTensor [N] or numpy array
        self.images = images.cpu()
        # store as plain Python ints for consistency with MNIST
        if torch.is_tensor(labels):
            self.labels = labels.cpu().tolist()
        else:
            self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return (Tensor image, Python int label)
        return self.images[idx], int(self.labels[idx])

# =============== Utilities ===============

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist(root="./data"):
    train_ds = datasets.MNIST(root=root, train=True,  download=True, transform=transforms.ToTensor())
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    return train_ds, test_ds


def filter_indices_by_labels(dataset, label_set):
    s = set(label_set)
    idx = [i for i, (_, y) in enumerate(dataset) if int(y) in s]
    return idx


def make_task_loaders(train_ds, test_ds, label_set, train_bs, test_bs, train_size=None, test_size=None, seed=42):
    train_idx = filter_indices_by_labels(train_ds, label_set)
    test_idx  = filter_indices_by_labels(test_ds,  label_set)

    rng = random.Random(seed)

    if train_size is not None and train_size < len(train_idx):
        train_idx = rng.sample(train_idx, k=train_size)
    if test_size is not None and test_size < len(test_idx):
        test_idx = rng.sample(test_idx, k=test_size)

    tr_subset = Subset(train_ds, train_idx)
    ts_subset = Subset(test_ds,  test_idx)

    tr = DataLoader(tr_subset, batch_size=train_bs, shuffle=True)
    ts = DataLoader(ts_subset,  batch_size=test_bs, shuffle=False)

    return tr, ts, len(tr_subset), len(ts_subset)


def _save_df(arr2d, path, col_names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(arr2d, columns=col_names)
    df.to_csv(path, index=False)  # no index column


@torch.no_grad()
def _collect_feats_logits_labels(model, loader, device, apply_softmax=True):
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        z, _, f = model(x) # (logits, recon, features)
        if apply_softmax:
            z = torch.softmax(z, dim=1)               
        feats.append(f.detach().cpu())
        logits.append(z.detach().cpu())
        labels.append(y.detach().cpu())
    feats  = torch.cat(feats,  dim=0).numpy()
    logits = torch.cat(logits, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy().reshape(-1, 1)
    return feats, logits, labels


def _filter_top_fraction_per_class(scores: np.ndarray,
                                   labels: np.ndarray,
                                   keep_frac: float,
                                   num_classes: int) -> np.ndarray:
    """
    Keep top keep_frac rows per TRUE class based on scores[:, true_class].
    Returns boolean mask length N.
    """
    y = labels.reshape(-1).astype(int)
    N = y.shape[0]
    mask = np.zeros(N, dtype=bool)

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        conf = scores[idx, c]                 # score of the TRUE class
        order = np.argsort(conf)[::-1]        # descending
        k = int(np.ceil(idx.size * keep_frac))
        k = max(1, k)                         # keep at least 1 if class exists
        keep_idx = idx[order[:k]]
        mask[keep_idx] = True

    return mask


@torch.no_grad()
def export_task_csvs(model, train_loader, list_test_loader, device, out_dir, f_size, num_classes=10, save_as="softmax", keep_frac=1.0):
    """
    Parameters:
     - save_y_as: "softmax" (default) or "logits" (raw outputs)
     - keep_frac [0, 1.0]: fraction of top samples to keep per class train-set based on confidence scores.

    Writes:
      - features: <out_dir>/train_feat.csv, <out_dir>/test_feat.csv   (f_size cols + 1 label)
    Returns dict of paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_softmax = (save_as == "softmax")

    if save_as not in ("softmax", "logits"):
        raise ValueError(f"Invalid save_as: {save_as}")
    
    if keep_frac < 0.0 or keep_frac > 1.0:
        raise ValueError(f"Invalid keep_frac: {keep_frac}")

    # Train feature CSV
    tr_f, tr_z, tr_y = _collect_feats_logits_labels(model, train_loader, device, apply_softmax=save_softmax)
    if keep_frac < 1.0:
        print(f"  - Filtering train: keep top {keep_frac*100:.1f}% per class based on returned scores.")
        filter_mask = _filter_top_fraction_per_class(tr_z, tr_y, keep_frac, num_classes)
        tr_f = tr_f[filter_mask]
        tr_z = tr_z[filter_mask]
        tr_y = tr_y[filter_mask]
    
    # column names
    score_prefix = "s" if save_softmax else "z" # s: softmax, z: logits
    feat_cols = [f"f{i}" for i in range(tr_f.shape[1])] + [f"z{i}" for i in range(tr_z.shape[1])] + ["label"]
    
    train_feat_csv = os.path.join(out_dir, "train_feat.csv")
    _save_df(np.concatenate([tr_f, tr_z, tr_y], axis=1), train_feat_csv, feat_cols)
    
    # Test CSV
    ts_feats, ts_logits, ts_labels = [], [], []
    for loader in list_test_loader:
        f, z, y = _collect_feats_logits_labels(model, loader, device, apply_softmax=save_softmax)
        ts_feats.append(f)
        ts_logits.append(z)
        ts_labels.append(y)

    ts_f = np.concatenate(ts_feats, axis=0)
    ts_z = np.concatenate(ts_logits, axis=0)
    ts_y = np.concatenate(ts_labels, axis=0)

    test_feat_csv = os.path.join(out_dir, "test_feat.csv")
    _save_df(np.concatenate([ts_f, ts_z, ts_y], axis=1), test_feat_csv, feat_cols)

    return {
        "train_feat_csv": train_feat_csv,
        "test_feat_csv":  test_feat_csv
    }


def save_checkpoint(out_dir, task_id, model):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_task{task_id}.pt")
    torch.save(model.state_dict(), path)
    print(f"[Task {task_id}] Checkpoint saved -> {path}")


def init_metrics_csv(out_dir, tasks, fname="metrics.csv"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    headers = [
        "timestamp",
        "task_id",
        "task_digits",
        "epoch_or_final",
        "stage",
        "train_loss_ae",
        "train_loss_head",
        "train_ce_head",
        "train_rec_ae",
        "train_feat_reg_ae",
        "train_logit_reg_head",
        "train_acc",
        "test_ce_loss", 
        "test_acc_mean",
        "test_accs_seen",
        "test_rec_loss",
    ]

    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
    return path, headers


def log_metrics_row(csv_path, headers, row_dict):
    row = [row_dict.get(h, "") for h in headers]
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)



def eval_seen_per_task(model, loaders_list, device):
    """
    Returns list of accuracies on tasks [0..t] using each task's test loader.
    Uses existing `test(...)` as-is (no masking).
    """
    accs = []
    for ld in loaders_list:
        res = test(model, ld, device=device, report_recon=False)
        accs.append(res["accuracy"])
    return accs

def eval_seen_mean(model, loaders_list, device):
    """
    Average accuracy across the provided test loaders (simple mean).
    Uses existing `test(...)` as-is (no masking).
    """
    accs = eval_seen_per_task(model, loaders_list, device)
    return float(np.mean(accs)) if len(accs) > 0 else 0.0

@torch.no_grad()
def evaluate_across_seen_tasks(model, seen_test_loaders, device):
    """
    Returns list of accuracies on tasks [0..t] using each task's test loader.
    """
    model.eval(); model.to(device)
    accs = []
    for ts in seen_test_loaders:
        correct, total = 0, 0
        for x, y in ts:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        accs.append(correct / max(total, 1))
    return accs


def pretty_task(tasks, t):
    return "[" + ",".join(map(str, tasks[t])) + "]"

def run_rscript_and_wait(rscript_path, r_args_str=None, cwd=None, r_cmd=None):
    """
    Runs: <Rscript command> <rscript_path> [r_args...]

    - On Windows:
        * If r_cmd is provided, it will be used (recommended).
        * Otherwise we try to find Rscript on PATH; if not found, we raise with a helpful message.
    - On macOS/Linux:
        * Uses 'Rscript' by default, unless r_cmd is provided.

    r_args_str is split with shlex.split so you can pass 'key=val key2=val2'.
    Raises on nonzero exit. Returns stdout (string).
    """
    # Normalize rscript_path and cwd
    rscript_path = str(rscript_path)
    if cwd is not None:
        cwd = str(Path(cwd))
        if not os.path.isdir(cwd):
            raise FileNotFoundError(f"cwd does not exist: {cwd}")

    system = platform.system()

    # Decide which Rscript command to use
    if r_cmd:
        r_exec = str(r_cmd)
        if not os.path.exists(r_exec):
            raise FileNotFoundError(f"r_cmd does not exist: {r_exec}")
    else:
        if system == "Windows":
            # Try PATH first
            r_exec = shutil.which("Rscript")
            if not r_exec:
                # Not found — ask user to supply r_cmd explicitly
                raise FileNotFoundError(
                    "Rscript not found on PATH. On Windows, pass r_cmd with the full path to Rscript.exe, "
                    "e.g., r_cmd=r'C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe'."
                )
        else:
            # macOS/Linux default
            r_exec = shutil.which("Rscript") or "Rscript"

    # Build command
    cmd = [r_exec, rscript_path]
    if isinstance(r_args_str, str):
        cmd += shlex.split(r_args_str, posix=(os.name != "nt"))
    elif isinstance(r_args_str, (list, tuple)):
        cmd += list(map(str, r_args_str))

    # Run
    proc = subprocess.run(cmd, cwd=cwd, stdout=None, stderr=None, check=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed with code {proc.returncode}\n"
            f"CMD: {cmd}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    return proc.stdout


def load_features_labels_from_Rcsv(csv_path, f_size, dtype=torch.float32, device="cpu"):
    """
    Expects CSV with exactly f_size feature columns (first) + 1 label column (last).
    Returns (features[N,f_size], labels[N]).
    """
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1) # gp output create headers
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != f_size + 1:
        raise ValueError(f"CSV shape {arr.shape} != (N, f_size+1) with f_size={f_size}")
    feats = torch.tensor(arr[:, :f_size], dtype=dtype, device=device)
    labels = torch.tensor(arr[:, f_size].astype(np.int64), dtype=torch.long, device=device)
    return feats, labels


# --- decode features back to images using your AE decoder ---
def decode_features_to_images(model, feats, batch_size=256, clamp_range=(0.0, 1.0), device=None):
    """
    Decode latent features [N, f_size] into images [N,1,28,28] using model.decode.
    """
    if device is None:
        device = feats.device
    model.eval()
    model.to(device)

    imgs = []
    for i in range(0, feats.size(0), batch_size):
        z = feats[i:i+batch_size].to(device)
        xhat = model.decode(z)               
        if xhat.dim() == 2:  # safety reshaping
            xhat = xhat.view(-1, 1, 28, 28)
        if clamp_range is not None:
            xhat = torch.clamp(xhat, clamp_range[0], clamp_range[1])
        imgs.append(xhat.detach().cpu())
    return torch.cat(imgs, dim=0)


def make_recovered_dataset(images, labels):
    return RecoveredDataset(images, labels)


def plot_acc_over_all_tasks(
    histories_per_task,
    epochs_per_task,
    title_prefix="Accuracy over Global Epochs",
    save_path_prefix=None
):
    num_tasks = len(histories_per_task)

    # ----- Compute global offsets -----
    offsets = [0]
    for e in epochs_per_task[:-1]:
        offsets.append(offsets[-1] + e)

    global_last_epoch = sum(epochs_per_task)

    # ==========================================================
    # TRAINING FIGURE
    # ==========================================================
    plt.figure(figsize=(10, 5))

    for t, hist in enumerate(histories_per_task):
        start = offsets[t]
        T = len(hist.get("train_acc", []))
        x = [start + i + 1 for i in range(T)]
        plt.plot(
            x,
            hist["train_acc"],
            linestyle="-",
            linewidth=2,
            label=f"Task {t}"
        )

    plt.xlabel("Global Epoch Index")
    plt.ylabel("Training Accuracy (%)")
    plt.title(f"{title_prefix} (Training)")
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2, frameon=False)

    if save_path_prefix is not None:
        plt.savefig(f"{save_path_prefix}_train.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()


    # ==========================================================
    # TEST FIGURE
    # ==========================================================
    plt.figure(figsize=(10, 5))

    max_tasks = num_tasks
    xs = [[] for _ in range(max_tasks)]
    ys = [[] for _ in range(max_tasks)]

    for t, hist in enumerate(histories_per_task):
        if "test_accs_seen" not in hist:
            continue

        start = offsets[t]
        test_list = hist["test_accs_seen"]

        for ep_idx, accs_seen in enumerate(test_list):
            if accs_seen is None:
                continue

            xg = start + ep_idx + 1
            for k, acc_k in enumerate(accs_seen):
                xs[k].append(xg)
                ys[k].append(acc_k)

    for k in range(max_tasks):
        if len(xs[k]) == 0:
            continue

        plt.plot(
            xs[k],
            ys[k],
            linestyle="--",
            linewidth=2,
            label=f"Task {k}"
        )

    plt.xlabel("Global Epoch Index")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"{title_prefix} (Test)")
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2, frameon=False)

    if save_path_prefix is not None:
        plt.savefig(f"{save_path_prefix}_test.png", bbox_inches="tight", dpi=300)
    else:
        plt.show()



def plotGP_per_class_metrics_over_tasks(
    run_dir,
    task_ids,
    metrics=("accuracy", "precision", "recall", "f1"),
    filename="gp_test_metrics.csv",
    title="Per-class metrics over tasks",
    save_path=None
):
    """
    For each task i, reads: {run_dir}/task{i}/{filename}
    Plots subplots for metrics; x=task, y=metric value.
    One line per class with consistent color across all subplots.
    Classes may differ across tasks (missing points are skipped).
    """
    # ---- read all tasks into one df ----
    dfs = []
    for t in task_ids:
        path = f"task{t}/{filename}"
        fullpath = os.path.join(run_dir, path)
        df = pd.read_csv(fullpath)
        df["task"] = int(t)
        # enforce numeric (robust to NA strings)
        df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
        for m in metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=["class"])
    all_df["class"] = all_df["class"].astype(int)


    # ---- consistent color per class ----
    classes = sorted(all_df["class"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(classes)}

    # ---- subplot layout ----
    metrics = [m for m in metrics if m in all_df.columns]
    if not metrics:
        raise ValueError("None of the requested metrics columns exist in the CSVs.")

    n = len(metrics)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), sharex=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    # ---- plot each metric, one line per class ----
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for c in classes:
            sub = all_df[all_df["class"] == c][["task", metric]].dropna()
            if sub.empty:
                continue
            sub = sub.sort_values("task")
            ax.plot(
                sub["task"],
                sub[metric],
                marker="o",
                label=f"class {c}",
                color=color_map[c],
            )
        ax.set_title(metric)
        ax.set_ylabel(f"{metric}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(task_ids))

    # hide unused axes if any
    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    for ax in axes[:len(metrics)]:
        ax.set_xlabel("Task")

    # single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plotGP_total_accuracy_over_tasks(
    run_dir,
    task_ids,
    filename="gp_test_metrics.csv",
    title="Total test accuracy over tasks",
    save_path=None
):
    """
    Reads total accuracy per task from {run_dir}/task{i}/{filename}.
    Assumes column 'total_accuracy' exists and is constant within each CSV.
    Plots x=task, y=total_accuracy.
    """
    xs, ys = [], []
    for t in task_ids:
        path = f"task{t}/{filename}"
        fullpath = os.path.join(run_dir, path)
        df = pd.read_csv(fullpath)

        if "total_accuracy" not in df.columns:
            raise ValueError(f"{fullpath} missing 'total_accuracy' column.")

        # same for all rows; take first non-NA
        val = pd.to_numeric(df["total_accuracy"], errors="coerce").dropna()
        if val.empty:
            raise ValueError(f"{fullpath} has no valid numeric total_accuracy.")
        total_acc = float(val.iloc[0])

        xs.append(int(t))
        ys.append(total_acc)

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xticks(list(task_ids))
    plt.xlabel("Task")
    plt.ylabel(f"Total accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()



def extract_head_only_history(history_2stage):
    """
    Keep ONLY the epochs where stage == "Head",
    preserving all keys consistently.
    """

    mask = [s == "Head" for s in history_2stage.get("stage", [])]

    head_hist = {}

    for key, values in history_2stage.items():
        # Only process list-like entries with same length as stage
        if isinstance(values, list) and len(values) == len(mask):
            head_hist[key] = [v for v, m in zip(values, mask) if m]
        else:
            # keep non-epoch keys untouched (rare case)
            head_hist[key] = values

    return head_hist

# =============== Main Driver ===============

def main():
    parser = argparse.ArgumentParser(description="Continual MNIST driver with NN_AE Model")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root",  type=str, default="./runs_mnist_continual")
    parser.add_argument("--gp_out",    type=str, default="gp_data")
    parser.add_argument("--ckpt_out",  type=str, default="checkpoints")
    parser.add_argument("--f_size",    type=int, default=16, help="feature dim after adapter")
    parser.add_argument("--epochs0",   type=int, default=60,  help="epochs for task 0")
    parser.add_argument("--epochs",    type=int, default=30,  help="epochs for tasks 1..")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--lr_AE",     type=float, default=1e-4)
    parser.add_argument("--lr_head",   type=float, default=1e-3)
    parser.add_argument("--train_bs",  type=int, default=128)
    parser.add_argument("--test_bs",   type=int, default=128)
    parser.add_argument("--lambda_rec",  type=float, default=0.5, help="weight for reconstruction loss")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="weight for feature preservation")
    parser.add_argument("--lambda_logit", type=float, default=1.0, help="weight for logit preservation")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--rscript_path", type=str, default="GP_train.R",
                        help="Path to the R script to run (used if --run_rscript).")
    parser.add_argument("--max_cache_items", type=int, default=None,
                        help="limit cached feature dict size (None = all)")
    parser.add_argument("--log_every_epoch", action="store_true",
                        help="also dump a CSV row each epoch (not just end-of-task)")
    parser.add_argument("--GP_train_size_per_class", type=int, default=2000)
    parser.add_argument("--GP_test_size_per_class", type=int, default=1000)
    parser.add_argument("--GP_train_otc_size", type=int, default=50)
    parser.add_argument("--GP_num_indcpts", type=int, default=1000)
    parser.add_argument("--GP_package", type=str, default="gplite")
    parser.add_argument("--skip_GP", action="store_true", help="Whether to skip the GP training R script after each task.") # for experiments convenience;
    parser.add_argument("--ce_onall", action="store_true", help="Whether to compute CE loss on all seen tasks' data (instead of just current task) during training.") # for experiments convenience;

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}; f_size={args.f_size}")

    skip_GP = args.skip_GP

    # Tasks: Task0 = [0..4], then single-digit tasks 5..9
    tasks = [[0,1],[2,3],[4,5],[6,7]]
    seen_classes_per_task = [sorted(sum(tasks[:t+1], [])) for t in range(len(tasks))]
    train_size = [15000,6000,6000,6000] #NOTE: Match number of tasks
    # tasks = [[0,1,2,3,4,5,6,7,8], [9]]

    # Output dirs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, f"run_{stamp}")
    gp_dir  = os.path.join(run_dir, args.gp_out)
    ck_dir  = os.path.join(run_dir, args.ckpt_out)
    os.makedirs(run_dir, exist_ok=True)

    # Persist config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # CSV logger
    csv_path, headers = init_metrics_csv(run_dir, tasks)

    # Data
    train_ds, test_ds = load_mnist(root=args.data_root)

    train_loaders, test_loaders = [], []
    for t, labels in enumerate(tasks):
        tr, ts, _, _ = make_task_loaders(train_ds, test_ds, labels, args.train_bs, args.test_bs, train_size=train_size[t], seed=args.seed)
        train_loaders.append(tr)
        test_loaders.append(ts)

    # Model
    model = CALM_AE_NN(f_size=args.f_size, num_classes=10).to(device)

    # --- No longer needed using train_2_stage_label_aware ---
    # # For feature-preservation, cache features after each task
    # old_feature_dict = None
    # old_logit_dict = None
    
    seen_test_loaders = []

    # Per Epoch Evaluation
    histories_per_task = []
    epochs_per_task = []
    test_end_accs = []


    # ------------- Loop over tasks -------------
    for t, tr_loader in enumerate(train_loaders):
        print("\n" + "="*80)
        print(f"Starting Task {t}: digits {tasks[t]}")
        print(f"Training samples size: {len(tr_loader.dataset)}")
        print("="*80)

        # Train
        # epochs = args.epochs0 if t == 0 else (args.epochs)*t  # train more epochs as tasks coming
        epochs = args.epochs0 if t == 0 else args.epochs  # train more epochs as tasks coming

        start_time = time.time()
        
        seen_test_loaders.append(test_loaders[t])
        # eval_fn = (lambda m: eval_seen_mean(m, seen_test_loaders, device)) if args.log_every_epoch else None
        # track per-task instead of mean:
        eval_fn = (lambda m: eval_seen_per_task(m, seen_test_loaders, device)) if args.log_every_epoch else None


        # history = train_2_stage(
        #     model=model,
        #     trloader=tr_loader,
        #     # --- Stage 1 (AE) ---
        #     epochs_stage1=epochs,
        #     lr_stage1=args.lr_AE,
        #     lambda_rec_stage1=args.lambda_rec,     # recon loss weight
        #     lambda_feat_stage1=(0.0 if t == 0 else args.lambda_feat),    # feature preservation weight (MSE(f_curr, f_prev))
        #     old_feature_dict=(None if t == 0 else old_feature_dict),     # {image_bytes: f_prev_tensor}
        #     # --- Stage 2 (Head) ---
        #     epochs_stage2=epochs,
        #     lr_stage2=args.lr_head,
        #     lambda_ce_stage2=1.0,      # CE weight
        #     lambda_logit_stage2=(0.0 if t == 0 else args.lambda_logit),   # logit preservation weight (MSE(z_curr, z_prev[mask]))
        #     old_logit_dict=(None if t == 0 else old_logit_dict),       # {image_bytes: z_prev_tensor} (prev logits)
        #     logit_mask=None,           # torch.BoolTensor[num_classes] or None
        #     # --- common ---
        #     device=device,
        #     grad_clip=None,
        #     eval_fn=eval_fn               # callable(model)->acc in [0,1]
        # )
        
        # ---- For train_2_stage_label_aware ---------
        # Build old/new class sets for this task (global class ids)
        old_classes = sum(tasks[:t], [])      # all digits seen before task t (flatten to a list)
        new_classes = tasks[t]               # digits introduced at task t

        # Teacher snapshot BEFORE training this task
        teacher_model = deepcopy(model).to(device).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Optional: preserve only OLD logits (recommended)
        # logit_mask = torch.zeros(10, dtype=torch.bool, device=device)
        # if len(old_classes) > 0:
        #     logit_mask[torch.tensor(old_classes, dtype=torch.long, device=device)] = True
        # else:
        #     logit_mask = None
        


        history = train_2_stage_class_aware(
            model=model,
            teacher_model=teacher_model,
            trloader=tr_loader,
            old_classes=old_classes,
            new_classes=new_classes,

            # -------- Stage 1 (AE) --------
            epochs_stage1=epochs,
            lr_stage1=args.lr_AE,
            lambda_rec=args.lambda_rec,
            lambda_feat=(0.0 if t == 0 else args.lambda_feat),
            rec_on="all",                 #Fixed

            # -------- Stage 2 (Head) --------
            epochs_stage2=epochs,
            lr_stage2=args.lr_head,
            lambda_ce=1.0,
            lambda_logit=(0.0 if t == 0 else args.lambda_logit),
            ce_on="all" if args.ce_onall else "new",                  # CE only on new samples or all seen samples
            phi_fill="mean",       

            # -------- Common --------
            device=device,
            grad_clip=None,
            eval_fn=eval_fn
        )
        # ----------------------------------------------


        dur = time.time() - start_time
        print(f"[Task {t}] Training done in {dur/60.0:.2f} min.")

        # Evaluate on current task test set
        res = test(model, test_loaders[t], device=device, report_recon=True)
        print(f"[Task {t}] Test (on new class only) -> ce_loss: {res['ce_loss']:.4f}, acc: {100*res['accuracy']:.2f}%, "
              f"rec_loss: {res.get('rec_loss', float('nan')):.4f}")
        
        histories_per_task.append(history)
        epochs_per_task.append(epochs)
        # For 2 stage train:
        head_histories_per_task = [extract_head_only_history(h) for h in histories_per_task]
        epochs_head_per_task = [len(h["train_acc"]) for h in head_histories_per_task]
        test_end_accs.append(res["accuracy"])


        # Prepare for GP
        task_dir = os.path.join(run_dir, f"task{t}")
        csv_paths = export_task_csvs(
            model=model,
            train_loader=tr_loader,
            list_test_loader=seen_test_loaders,
            device=device,
            out_dir=task_dir,
            f_size=args.f_size,
            num_classes=10,
            save_as="softmax", #FIXME: change it to softmax later
            keep_frac=0.95 #NOTE: keep top 95% confident samples per class (for GP training)
        )
        print(f"[Task {t}] Wrote CSVs to: {task_dir}")
        data_tr_path = csv_paths["train_feat_csv"]   
        data_ts_path = csv_paths["test_feat_csv"]  

        # Evaluate across all seen tasks
        accs_seen = evaluate_across_seen_tasks(model, seen_test_loaders, device)
        mean_seen = float(np.mean(accs_seen)) if len(accs_seen) > 0 else float("nan")
        print("Accuracies on seen tasks:")
        for i, a in enumerate(accs_seen):
            print(f"  Task {i} {pretty_task(tasks,i)}: {100*a:.2f}%")
        print(f"Mean over seen tasks (0..{t}): {100*mean_seen:.2f}%")

        # CSV logging: end-of-task row        
        # 2-Satge Version:
        idx_head = max(i for i,s in enumerate(history["stage"]) if s == "Head")
        idx_ae   = max(i for i,s in enumerate(history["stage"]) if s == "AE")
        
        row_end = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": t,
            "task_digits": str(tasks[t]),
            "epoch_or_final": f"final_e{epochs}",
            "stage": "Final",
            "train_loss_ae": history["loss"][idx_ae],
            "train_loss_head": history["loss"][idx_head],
            "train_ce_head": history["ce"][idx_head],
            "train_logit_reg_head": history["logit_reg"][idx_head],
            "train_rec_ae": history["rec"][idx_ae],
            "train_feat_reg_ae": history["feat_reg"][idx_ae],
            "train_acc": history["train_acc"][idx_head],
            "test_ce_loss": res["ce_loss"],
            "test_rec_loss": res.get("rec_loss", ""),
            "test_acc_mean": mean_seen,
            "test_accs_seen": accs_seen
        }
        # for i in range(len(tasks)):
        #     row_end[f"acc_task{i}"] = (accs_seen[i] if i < len(accs_seen) else "")
        log_metrics_row(csv_path, headers, row_end)

        # Optional per-epoch logging rows
        if args.log_every_epoch:
            for ep_idx in range(len(history["loss"])):
                stage_tag = history["stage"][ep_idx] if "stage" in history else ""
                row_ep = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": t,
                    "task_digits": str(tasks[t]),
                    "epoch_or_final": f"epoch_{ep_idx+1}_{stage_tag}",
                    "stage": stage_tag,
                    "train_loss_ae": history["loss"][ep_idx] if stage_tag == "AE" else "",
                    "train_loss_head": history["loss"][ep_idx] if stage_tag == "HEAD" else "",
                    "train_ce_head": history["ce"][ep_idx] if stage_tag == "Head" else "", # Head classification CE
                    "train_rec_ae": history["rec"][ep_idx] if stage_tag == "AE" else "", # AE recon loss
                    "train_feat_reg_ae": history["feat_reg"][ep_idx] if stage_tag == "AE" else "", # AE feature preservation
                    "train_logit_reg_head": history["logit_reg"][ep_idx] if stage_tag == "Head" else "", # Head logit preservation
                    "train_acc": history["train_acc"][ep_idx], # this is overall train acc (training classification accuracy average of seen classes)
                    "test_acc_mean": history.get("test_acc_mean", [None])[ep_idx],
                    "test_accs_seen": history.get("test_accs_seen", [None])[ep_idx],
                    "test_ce_loss": "", # I didn't track this per epoch
                    "test_rec_loss": "",
                }
                log_metrics_row(csv_path, headers, row_ep)

        # Checkpoint
        if not args.no_checkpoint:
            save_checkpoint(ck_dir, t, model)

        if skip_GP:
            print(f"[Task {t}] Skipping GP training and Rscript execution as per --skip_GP flag.")
            # prepare inducing_points.csv with just the original training features for next task (no GP selection)
            df_trGP = pd.read_csv(data_tr_path)
            n_seen = len(seen_classes_per_task[t])
            df_trGP = df_trGP.sample(n=min(args.GP_num_indcpts * n_seen, len(df_trGP)), random_state=args.seed)  # random subset of train_feat.csv of size=GP_num_indcpts*(num_seen_classes)
            df_trGP = df_trGP.iloc[:, list(range(args.f_size)) + [-1]]  # only first f_size columns  + label column (assumes label is last column), skip whaterver in middle
            df_trGP.to_csv(os.path.join(task_dir, "inducing_points.csv"), index=False)
        
        else:
            # Run Rscript and reconstruct R inducing points as "buffer" for next task
            print(f"[Task {t}] Running Rscript to produce: {task_dir}/inducing_points.csv")
            r_args = [
                "--n_tr", str(args.GP_train_size_per_class),
                "--n_ts", str(args.GP_test_size_per_class),
                "--n_octr", str(args.GP_train_otc_size),
                "--n_indcpts", str(args.GP_num_indcpts),
                "--GP_package", args.GP_package,
                "--save_path", str(task_dir),
                "--data_tr", str(data_tr_path),
                "--data_ts", str(data_ts_path),
                "--existing_classes", ",".join(str(c) for c in seen_classes_per_task[t]),
            ]
            
            if platform.system() == "Windows":
                run_rscript_and_wait(
                    rscript_path=Path.cwd() / args.rscript_path,
                    r_args_str=r_args,
                    cwd=Path.cwd(),
                    r_cmd=r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe",  # <-- Windows override
                )
            else:
                run_rscript_and_wait(args.rscript_path, r_args_str=r_args, cwd=None)

        R_inducing_path = os.path.join(task_dir, "inducing_points.csv")
        if not os.path.exists(R_inducing_path):
            raise FileNotFoundError(f"Expected R output CSV not found: {R_inducing_path}")
        
        # --- Prepare for next task ---
        if t+1 < len(train_loaders):
            # Load features+labels back from CSV
            feats_csv, labels_csv = load_features_labels_from_Rcsv(
                R_inducing_path, f_size=args.f_size, dtype=torch.float32, device=device
            )
            print(f"[Task {t}] Loaded from R CSV: {feats_csv.shape[0]} rows, f_size={feats_csv.shape[1]}")

            # Decode to images with AE decoder
            recovered_imgs = decode_features_to_images(
                model, feats_csv, batch_size=128, clamp_range=(0.0, 1.0), device=device
            )
            recovered_ds = make_recovered_dataset(recovered_imgs, labels_csv)
            print(f"[Task {t}] Decoded images: {recovered_imgs.shape} -> recovered dataset size {len(recovered_ds)}")

            print(f"[Task {t}] Augmenting Task {t+1} training set with {len(recovered_ds)} recovered samples.")
            base_ds_next = train_loaders[t+1].dataset  # usually a Subset
            aug_ds_next = ConcatDataset([base_ds_next, recovered_ds])

            # Rebuild the next task's DataLoader with the SAME params
            train_loaders[t+1] = DataLoader(
                aug_ds_next,
                batch_size=args.train_bs,
                shuffle=True,            # keep shuffle=True for training
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )


            # # Build/update feature dict AFTER task t (to preserve seen features next task)
            # print(f"[Task {t}] Caching f_size features for feature-preservation in next task...")
            # cache_loader = DataLoader(Subset(train_ds, filter_indices_by_labels(train_ds, sum(tasks[:t+1], []))),
            #                         batch_size=args.test_bs, shuffle=False, num_workers=2, pin_memory=True)
            # old_feature_dict, old_logit_dict = build_feature_dict(
            #     model=model,
            #     dataloader=cache_loader,
            #     device=device,
            #     max_items=args.max_cache_items,
            #     dtype=torch.float32
            # )
            # print(f"[Task {t}] Cached features: {len(old_feature_dict)} items.")

    print("\nAll tasks complete.")
    print(f"Run directory: {run_dir}")
    print(f"CSV metrics:   {csv_path}")
    print(f"GP training files:    {gp_dir}")
    if not args.no_checkpoint:
        print(f"Checkpoints:   {ck_dir}")


    # Below is for 2-stage train, head-only (not include AE) accuracy plot
    plot_acc_over_all_tasks(
        histories_per_task=head_histories_per_task,
        epochs_per_task=epochs_head_per_task,
        title_prefix="MNIST Continual: NN (Head) Accuracy vs Global Epochs",
        save_path_prefix=os.path.join(run_dir, "acc_over_time_head_only")
    )

    if not skip_GP:
        # Plot GP metrics
        plotGP_per_class_metrics_over_tasks(
            run_dir=run_dir,
            task_ids=list(range(len(tasks))),
            metrics=("accuracy", "precision", "recall", "f1"),
            save_path=os.path.join(run_dir, "gp_per_class_metrics.png")
        )

        plotGP_total_accuracy_over_tasks(
            run_dir=run_dir,
            task_ids=list(range(len(tasks))),
            save_path=os.path.join(run_dir, "gp_total_accuracy.png")
        )


if __name__ == "__main__":
    main()
