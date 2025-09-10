# run_continual_mnist_driver.py
import os
import csv
import json
import time
import math
import random
import argparse
from datetime import datetime


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
    headers = ["timestamp","task_id","task_digits","epoch_or_final","train_loss","train_ce","train_rec","train_feat_reg",
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
    parser.add_argument("--max_cache_items", type=int, default=None,
                        help="limit cached feature dict size (None = all)")
    parser.add_argument("--log_every_epoch", action="store_true",
                        help="also dump a CSV row each epoch (not just end-of-task)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}; f_size={args.f_size}")

    # Tasks: Task0 = [0..4], then single-digit tasks 5..9
    tasks = [[0,1,2,3,4],[5],[6],[7],[8],[9]]
    tasks = [[0,1,2,3,4,5,6,7,8], [9]]

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
    seen_test_loaders = []

    # ------------- Loop over tasks -------------
    for t, tr_loader in enumerate(train_loaders):
        print("\n" + "="*80)
        print(f"Starting Task {t}: digits {tasks[t]}")
        print("="*80)

        # Freeze head for tasks after 0
        freeze_flag = False
        if t == 0:
            freeze_flag = False
            print("Task 0: head is UNFROZEN (learning initial classifier).")
        else:
            freeze_flag = True
            print("Task >0: FREEZING classifier head (fc2+fc3).")
            

        # Train
        epochs = args.epochs0 if t == 0 else args.epochs
        start_time = time.time()

        history = train(
            model=model,
            trloader=tr_loader,
            epochs=epochs,
            lr=args.lr,
            lambda_rec=args.lambda_rec,
            lambda_feat=(0.0 if t == 0 else args.lambda_feat),
            old_feature_dict=(None if t == 0 else old_feature_dict),
            freeze_head=freeze_flag,
            device=device,
            grad_clip=None
        )

        dur = time.time() - start_time
        print(f"[Task {t}] Training done in {dur/60.0:.2f} min.")

        # Save csv for GP training (features/logits/labels)
        Z, LOG, Y = extract_features_and_logits(model, tr_loader, device)
        save_gp_csv(gp_dir, t, Z, LOG, Y)

        # Evaluate on current task test set
        res = test(model, test_loaders[t], device=device, report_recon=True)
        print(f"[Task {t}] Test -> ce_loss: {res['ce_loss']:.4f}, acc: {100*res['accuracy']:.2f}%, "
              f"rec_loss: {res.get('rec_loss', float('nan')):.4f}")

        # Evaluate across all seen tasks
        seen_test_loaders.append(test_loaders[t])
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

        # Build/update feature dict AFTER task t (to preserve seen features next task)
        print(f"[Task {t}] Caching f_size features for feature-preservation in next task...")
        cache_loader = DataLoader(Subset(train_ds, filter_indices_by_labels(train_ds, sum(tasks[:t+1], []))),
                                  batch_size=args.test_bs, shuffle=False, num_workers=2, pin_memory=True)
        old_feature_dict = build_feature_dict(
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


if __name__ == "__main__":
    main()
