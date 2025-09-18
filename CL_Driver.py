# run_continual_mnist_driver.py
import os
import csv
import json
import time
import math
import random
import argparse
from datetime import datetime
import re
import matplotlib.pyplot as plt
import os, subprocess, shlex

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset, Dataset
from torchvision import datasets, transforms


# ---- Adapt these imports to your filenames/modules ----
# Assumes your model returns (logits, recon, f) where f is f_size features
from NN_AE import CALM_AE_NN
from NN_AE_utils import (
    train,  # CE + λ_rec*MSE(recon,x) + λ_feat*MSE(f,f_prev)
    test,                             # eval CE (and optional recon report)
    build_feature_dict,               # caches {image_bytes -> f_size feature}
    freeze_classifier_head,           # freezes fc2+fc3
    unfreeze_classifier_head
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
def _collect_feats_logits_labels(model, loader, device):
    model.eval()
    feats, logits, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        z, _, f = model(x)              # (logits, recon, features)
        feats.append(f.detach().cpu())
        logits.append(z.detach().cpu())
        labels.append(y.detach().cpu())
    feats  = torch.cat(feats,  dim=0).numpy()
    logits = torch.cat(logits, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy().reshape(-1, 1)
    return feats, logits, labels

@torch.no_grad()
def export_task_csvs(model, train_loader, test_loader, device, out_dir, f_size, num_classes=10):
    """
    Writes:
      - features: <out_dir>/train_feat.csv, <out_dir>/test_feat.csv   (f_size cols + 1 label)
    Returns dict of paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    tr_f, tr_z, tr_y = _collect_feats_logits_labels(model, train_loader, device)
    ts_f, ts_z, ts_y = _collect_feats_logits_labels(model, test_loader, device)

    # column names
    feat_cols = [f"f{i}" for i in range(tr_f.shape[1])] + [f"z{i}" for i in range(tr_z.shape[1])] + ["label"]

    # Save feature CSVs
    train_feat_csv = os.path.join(out_dir, "train_feat.csv")
    test_feat_csv  = os.path.join(out_dir, "test_feat.csv")
    _save_df(np.concatenate([tr_f, tr_z, tr_y], axis=1), train_feat_csv, feat_cols)
    _save_df(np.concatenate([ts_f, ts_z, ts_y], axis=1), test_feat_csv, feat_cols)

    return {
        "train_feat_csv": train_feat_csv,
        "test_feat_csv":  test_feat_csv
    }



@torch.no_grad()
def extract_features_and_logits(model, dataloader, device):
    """
    Returns:
        Z:   [N, f_size] features
        LOG: [N, 10] logits
        Y:   [N]
    """
    model.eval(); model.to(device)
    feats, logs, labs = [], [], []
    for x, y in dataloader:
        x = x.to(device)
        logits, recon, f = model(x)
        feats.append(f.detach().cpu())
        logs.append(logits.detach().cpu())
        labs.append(y.clone())
    return torch.cat(feats), torch.cat(logs), torch.cat(labs)


def save_gp_csv(out_dir, task_id, Z, LOG, Y):
    """
    Saves a CSV with columns: f_0..f_{f_size-1}, logit_0..logit_9, label
    Used for training GP models
    """
    os.makedirs(out_dir, exist_ok=True)

    Z_np   = Z.detach().cpu().numpy()          # [N, f_size]
    LOG_np = LOG.detach().cpu().numpy()        # [N, 10]
    Y_np   = Y.detach().cpu().numpy().reshape(-1, 1)  # [N, 1]

    data = np.concatenate([Z_np, LOG_np, Y_np], axis=1)

    f_cols   = [f"f_{i}" for i in range(Z_np.shape[1])]
    log_cols = [f"logit_{i}" for i in range(LOG_np.shape[1])]
    cols = f_cols + log_cols + ["label"]

    df = pd.DataFrame(data, columns=cols)
    df["label"] = df["label"].astype(int)  # keep labels as ints

    path = os.path.join(out_dir, f"task{task_id}_features_logits.csv")
    df.to_csv(path, index=False)
    print(f"[Task {task_id}] Saved GP CSV -> {path} "
          f"(features {tuple(Z_np.shape)}, logits {tuple(LOG_np.shape)})")


def save_checkpoint(out_dir, task_id, model):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"model_task{task_id}.pt")
    torch.save(model.state_dict(), path)
    print(f"[Task {task_id}] Checkpoint saved -> {path}")


def init_metrics_csv(out_dir, tasks, fname="metrics.csv"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    headers = ["timestamp","task_id","task_digits","epoch_or_final","train_loss","train_ce","train_rec","train_feat_reg", "train_logit_reg", 
               "test_loss","test_acc","test_rec_loss","seen_tasks_mean_acc"] + \
              [f"acc_task{i}" for i in range(len(tasks))]
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
    return path, headers


def log_metrics_row(csv_path, headers, row_dict):
    row = [row_dict.get(h, "") for h in headers]
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def eval_seen_mean(model, loaders_list, device):
    """
    Average accuracy across the provided test loaders (simple mean).
    Uses existing `test(...)` as-is (no masking).
    """
    accs = []
    for ld in loaders_list:
        res = test(model, ld, device=device, report_recon=False)
        accs.append(res["accuracy"])
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

def run_rscript_and_wait(rscript_path, r_args_str=None, cwd=None):
    """
    Runs: Rscript <rscript_path> [r_args...]
    r_args_str is split with shlex.split so you can pass 'key=val key2=val2'
    Raises on nonzero exit or if file outputs are missing (you check existence after call).
    """
    cmd = ["Rscript", rscript_path]
    if r_args_str:
        cmd += shlex.split(r_args_str)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
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
    per_epoch_test_acc_available=True,
    test_end_accs=None,
    title="Training & Testing Accuracy over Global Epochs",
    save_path=None
):
    """
    histories_per_task: list of history dicts returned by train(), one per task.
                        Must include history["acc"] (per-epoch train acc, 0..100).
                        If per_epoch_test_acc_available, also include history["test_acc"] (0..100).
    epochs_per_task: list of ints, the epoch count used for each task (len == num_tasks).
    per_epoch_test_acc_available: bool, True if you enabled eval_fn during training.
    test_end_accs: list of floats 0..1 for end-of-task test accuracy (len == num_tasks).
    """
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
    # Build global x-axis offsets
    offsets = [0]
    for e in epochs_per_task[:-1]:
        offsets.append(offsets[-1] + e)

    plt.figure()
    global_last_epoch = 0

    for t, hist in enumerate(histories_per_task):
        start = offsets[t]
        T = len(hist.get("acc", []))
        x = [start + i + 1 for i in range(T)]  # epoch indices start at 1 per task

        # Training accuracy (solid)
        plt.plot(x, hist["acc"], label=f"Task {t} train")

        # Testing accuracy
        if per_epoch_test_acc_available and "test_acc" in hist:
            print("test per epoch")
            plt.plot(x, hist["test_acc"], linestyle="--", label=f"Task {t} test")
        else:
            # end-of-task marker if provided
            if test_end_accs is not None and t < len(test_end_accs):
                end_x = start + T
                plt.scatter([end_x], [100.0 * test_end_accs[t]], marker="o", label=f"Task {t} test (final)")

        global_last_epoch = max(global_last_epoch, start + T)

    plt.xlabel("Global Epoch Index")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

# =============== Main Driver ===============

def main():
    parser = argparse.ArgumentParser(description="Continual MNIST driver with NN_AE Model")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root",  type=str, default="./runs_mnist_continual")
    parser.add_argument("--gp_out",    type=str, default="gp_data")
    parser.add_argument("--ckpt_out",  type=str, default="checkpoints")
    parser.add_argument("--f_size",    type=int, default=16, help="feature dim after adapter")
    parser.add_argument("--epochs0",   type=int, default=30,  help="epochs for task 0")
    parser.add_argument("--epochs",    type=int, default=30,  help="epochs for tasks 1..")
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--train_bs",  type=int, default=128)
    parser.add_argument("--test_bs",   type=int, default=128)
    parser.add_argument("--lambda_rec",  type=float, default=0.5, help="weight for reconstruction loss")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="weight for feature preservation")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--rscript_path", type=str, default="GP_train.R",
                        help="Path to the R script to run (used if --run_rscript).")
    parser.add_argument("--max_cache_items", type=int, default=None,
                        help="limit cached feature dict size (None = all)")
    parser.add_argument("--log_every_epoch", action="store_true",
                        help="also dump a CSV row each epoch (not just end-of-task)")
    parser.add_argument("--GP_train_size_per_class", type=int, default=1000)
    parser.add_argument("--GP_test_size_per_class", type=int, default=1500)
    parser.add_argument("--GP_train_otc_size", type=int, default=50)
    parser.add_argument("--GP_num_indcpts", type=int, default=1000)
    parser.add_argument("--GP_package", type=str, default="gplite")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}; f_size={args.f_size}")

    # Tasks: Task0 = [0..4], then single-digit tasks 5..9
    tasks = [[0,1,2,3,4],[5],[6],[7],[8],[9]]
    last_digits = [4,5,6,7,8,9]
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
        tr, ts, _, _ = make_task_loaders(train_ds, test_ds, labels, args.train_bs, args.test_bs, seed=args.seed)
        train_loaders.append(tr)
        test_loaders.append(ts)

    # Model
    model = CALM_AE_NN(f_size=args.f_size, num_classes=10).to(device)

    # For feature-preservation, cache features after each task
    old_feature_dict = None
    old_logit_dict = None
    seen_test_loaders = []

    # Per Epoch Evaluation
    histories_per_task = []
    epochs_per_task = []
    test_end_accs = []


    # ------------- Loop over tasks -------------
    for t, tr_loader in enumerate(train_loaders):
        print("\n" + "="*80)
        print(f"Starting Task {t}: digits {tasks[t]}")
        print("="*80)

        # Freeze head for tasks after 0
        # freeze_flag = False
        # if t == 0:
        #     freeze_flag = False
        #     print("Task 0: head is UNFROZEN (learning initial classifier).")
        # else:
        #     freeze_flag = True
        #     print("Task >0: FREEZING classifier head (fc2+fc3).")

        # NOTE: Do not freeze 
        freeze_flag = False

        # Train
        epochs = args.epochs0 if t == 0 else args.epochs
        start_time = time.time()
        
        seen_test_loaders.append(test_loaders[t])
        eval_fn = (lambda m: eval_seen_mean(m, seen_test_loaders, device)) if args.log_every_epoch else None
        
        history = train(
            model=model,
            trloader=tr_loader,
            epochs=epochs,
            lr=args.lr,
            lambda_rec=args.lambda_rec,
            lambda_feat=(0.0 if t == 0 else args.lambda_feat),
            old_feature_dict=(None if t == 0 else old_feature_dict),
            old_logit_dict=(None if t == 0 else old_logit_dict),
            freeze_head=freeze_flag,
            device=device,
            grad_clip=None,
            eval_fn=eval_fn
        )

        dur = time.time() - start_time
        print(f"[Task {t}] Training done in {dur/60.0:.2f} min.")

        # # Save csv for GP training (features/logits/labels)
        # Z, LOG, Y = extract_features_and_logits(model, tr_loader, device)
        # save_gp_csv(gp_dir, t, Z, LOG, Y)

        # Evaluate on current task test set
        res = test(model, test_loaders[t], device=device, report_recon=True)
        print(f"[Task {t}] Test -> ce_loss: {res['ce_loss']:.4f}, acc: {100*res['accuracy']:.2f}%, "
              f"rec_loss: {res.get('rec_loss', float('nan')):.4f}")
        
        histories_per_task.append(history)
        epochs_per_task.append(epochs)
        test_end_accs.append(res["accuracy"])

        # Prepare for GP
        task_dir = os.path.join(run_dir, f"task{t}")
        csv_paths = export_task_csvs(
            model=model,
            train_loader=tr_loader,
            test_loader=test_loaders[t],
            device=device,
            out_dir=task_dir,
            f_size=args.f_size,
            num_classes=10
        )
        print(f"[Task {t}] Wrote CSVs to: {task_dir}")
        data_tr_path = csv_paths["train_feat_csv"]   # "filtered_train_sftmx.csv"
        data_ts_path = csv_paths["test_feat_csv"]  

        # Evaluate across all seen tasks
        accs_seen = evaluate_across_seen_tasks(model, seen_test_loaders, device)
        mean_seen = float(np.mean(accs_seen)) if len(accs_seen) > 0 else float("nan")
        print("Accuracies on seen tasks:")
        for i, a in enumerate(accs_seen):
            print(f"  Task {i} {pretty_task(tasks,i)}: {100*a:.2f}%")
        print(f"Mean over seen tasks (0..{t}): {100*mean_seen:.2f}%")

        # CSV logging: end-of-task row
        row_end = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task_id": t,
            "task_digits": str(tasks[t]),
            "epoch_or_final": f"final_e{epochs}",
            "train_loss": history["loss"][-1] if "loss" in history else "",
            "train_ce": history["ce"][-1] if "ce" in history else "",
            "train_rec": history["rec"][-1] if "rec" in history else "",
            "train_feat_reg": history["feat_reg"][-1] if "feat_reg" in history else "",
            "test_ce_loss": res["ce_loss"],
            "test_acc": res["accuracy"],
            "test_rec_loss": res.get("rec_loss", ""),
            "seen_tasks_mean_acc": mean_seen
        }
        for i in range(len(tasks)):
            row_end[f"acc_task{i}"] = (accs_seen[i] if i < len(accs_seen) else "")
        log_metrics_row(csv_path, headers, row_end)

        # Optional per-epoch logging rows
        if args.log_every_epoch:
            for ep_idx in range(len(history["loss"])):
                row_ep = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task_id": t,
                    "task_digits": str(tasks[t]),
                    "epoch_or_final": f"epoch_{ep_idx+1}",
                    "train_loss": history["loss"][ep_idx],
                    "train_ce": history["ce"][ep_idx],
                    "train_rec": history["rec"][ep_idx] if "rec" in history else "",
                    "train_feat_reg": history["feat_reg"][ep_idx] if "feat_reg" in history else "",
                    "train_logit_reg": history["logit_reg"][ep_idx] if "logit_reg" in history else "",
                    "train_acc": history["acc"][ep_idx],    
                    # test metrics per-epoch not computed to keep it fast
                    "test_ce_loss": "",
                    "test_acc": "",
                    "test_rec_loss": "",
                    "seen_tasks_mean_acc": ""
                }
                log_metrics_row(csv_path, headers, row_ep)

        # Checkpoint
        if not args.no_checkpoint:
            save_checkpoint(ck_dir, t, model)

        # Run Rscript and reconstruct R inducing points as "buffer" for next task
        print(f"[Task {t}] Running Rscript to produce: {task_dir}/inducing_points.csv")
        r_args = (
            f'--n_tr "{args.GP_train_size_per_class}" '
            f'--n_ts "{args.GP_test_size_per_class}" '
            f'--n_otcr "{args.GP_train_otc_size}" '
            f'--n_indcpts "{args.GP_num_indcpts}" '
            f'--GP_package "{args.GP_package}" '
            f'--save_path "{task_dir}" '
            f'--data_tr "{data_tr_path}" '
            f'--data_ts "{data_ts_path}" '
            f'--last_class "{last_digits[t]}"'
        )   
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

            # Decode to images with your AE decoder
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

            # Build/update feature dict AFTER task t (to preserve seen features next task)
            print(f"[Task {t}] Caching f_size features for feature-preservation in next task...")
            cache_loader = DataLoader(Subset(train_ds, filter_indices_by_labels(train_ds, sum(tasks[:t+1], []))),
                                    batch_size=args.test_bs, shuffle=False, num_workers=2, pin_memory=True)
            old_feature_dict, old_logit_dict = build_feature_dict(
                model=model,
                dataloader=cache_loader,
                device=device,
                max_items=args.max_cache_items,
                dtype=torch.float32
            )
            print(f"[Task {t}] Cached features: {len(old_feature_dict)} items.")

    print("\nAll tasks complete.")
    print(f"Run directory: {run_dir}")
    print(f"CSV metrics:   {csv_path}")
    print(f"GP training files:    {gp_dir}")
    if not args.no_checkpoint:
        print(f"Checkpoints:   {ck_dir}")

    plot_acc_over_all_tasks(
        histories_per_task=histories_per_task,
        epochs_per_task=epochs_per_task,
        per_epoch_test_acc_available=("test_acc" in histories_per_task[0]),
        test_end_accs=test_end_accs,
        title="MNIST Continual: Accuracy vs Global Epochs",
        save_path=os.path.join(run_dir, "acc_over_time.png")
    )

if __name__ == "__main__":
    main()
