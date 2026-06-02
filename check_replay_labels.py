"""
check_replay_labels.py

For each task in a run directory, loads the checkpoint model and evaluates
predicted labels on the replay_points.csv (GP-sampled feature points).
Optionally also evaluates the GP models on the same replay points (--gp_eval),
so you can see NN vs GP confusion side-by-side.

Usage:
    python3 check_replay_labels.py --run_dir runs/run_<stamp>
    python3 check_replay_labels.py --run_dir runs/run_<stamp> --ckpt_task 2
    python3 check_replay_labels.py --run_dir runs/run_<stamp> --tasks 1 2 3 --plot
    python3 check_replay_labels.py --run_dir runs/run_<stamp> --gp_eval

Arguments:
    --run_dir       Path to the run directory (contains config.json, checkpoints/, task*/)
    --ckpt_task     Use this task's checkpoint for ALL evaluations (default: same task as replay)
    --tasks         Which task directories to evaluate (default: all found)
    --batch_size    Batch size for inference (default: 512)
    --plot          Save confusion matrix PNGs per task (requires matplotlib)
    --gp_eval       Also run GP argmax on replay points via check_gp_on_replay.R
"""

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import platform

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from NN_AE import CALM_AE_NN


# ---------------------------------------------------------------------------- helpers

def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def load_checkpoint(run_dir, task_id, f_size, num_classes=10, device="cpu"):
    path = os.path.join(run_dir, "checkpoints", f"model_task{task_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = CALM_AE_NN(f_size=f_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_replay_points(task_dir, f_size):
    path = os.path.join(task_dir, "replay_points.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"replay_points.csv not found in {task_dir}")

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip().strip('"') for h in next(reader)]
        rows   = list(reader)

    feat_cols = [f"f{i}" for i in range(f_size)]
    missing   = [c for c in feat_cols if c not in header]
    if missing:
        raise ValueError(f"Missing feature columns in replay_points.csv: {missing}")
    if "label" not in header:
        raise ValueError("replay_points.csv has no 'label' column")

    feat_idx  = [header.index(c) for c in feat_cols]
    label_idx = header.index("label")

    feat_list, label_list = [], []
    for row in rows:
        if not row:
            continue
        feat_list.append([float(row[i]) for i in feat_idx])
        label_list.append(int(float(row[label_idx])))

    feats  = torch.tensor(feat_list,  dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.long)
    return feats, labels


def load_gp_preds_csv(path):
    """Read replay_gp_preds.csv written by check_gp_on_replay.R.
    Returns (true_labels list, pred_labels list).
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)
    true_labels = [int(float(r["true_label"])) for r in rows]
    pred_labels = [int(float(r["pred_label"])) for r in rows]
    return true_labels, pred_labels


@torch.no_grad()
def predict_from_features(model, feats, batch_size, device):
    model.eval()
    preds = []
    for i in range(0, feats.size(0), batch_size):
        batch  = feats[i : i + batch_size].to(device)
        logits = model.forward_from_adapter(batch)
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds)


def per_class_stats_torch(true_t, pred_t, classes):
    rows = []
    for c in classes:
        mask    = true_t == c
        n       = int(mask.sum().item())
        if n == 0:
            rows.append({"class": c, "n": 0, "correct": 0, "accuracy": float("nan")})
            continue
        correct = int((pred_t[mask] == c).sum().item())
        rows.append({"class": c, "n": n, "correct": correct, "accuracy": correct / n})
    return rows


def per_class_stats_list(true_list, pred_list, classes):
    rows = []
    for c in classes:
        n       = sum(1 for t in true_list if t == c)
        if n == 0:
            rows.append({"class": c, "n": 0, "correct": 0, "accuracy": float("nan")})
            continue
        correct = sum(1 for t, p in zip(true_list, pred_list) if t == c and p == c)
        rows.append({"class": c, "n": n, "correct": correct, "accuracy": correct / n})
    return rows


def confusion_matrix(true_list, pred_list, classes):
    idx = {c: i for i, c in enumerate(classes)}
    k   = len(classes)
    cm  = [[0] * k for _ in range(k)]
    for t, p in zip(true_list, pred_list):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def print_stats_block(label, n_total, n_correct, stats, wrong_true, wrong_pred, classes):
    overall = n_correct / n_total if n_total else float("nan")
    print(f"  [{label}] Overall accuracy: {n_correct}/{n_total} = {100*overall:.2f}%")
    print(f"  {'Class':>6}  {'N':>5}  {'Correct':>8}  {'Acc':>7}")
    for row in stats:
        acc_str = f"{100*row['accuracy']:.1f}%" if not math.isnan(row["accuracy"]) else "  n/a"
        print(f"  {row['class']:>6}  {row['n']:>5}  {row['correct']:>8}  {acc_str:>7}")
    n_wrong = len(wrong_true)
    if n_wrong > 0:
        print(f"\n  [{label}] Misclassifications ({n_wrong}):")
        for c in classes:
            pairs = [p for t_c, p in zip(wrong_true, wrong_pred) if t_c == c]
            if not pairs:
                continue
            counts = {}
            for p in pairs:
                counts[p] = counts.get(p, 0) + 1
            pred_str = ", ".join(f"{p}({cnt})" for p, cnt in sorted(counts.items()))
            print(f"    True={c} -> predicted as: {pred_str}")


def plot_confusion(cm, classes, title, save_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return
    k      = len(classes)
    flat   = [v for row in cm for v in row]
    cmax   = max(flat) if flat else 1
    thresh = cmax / 2.0
    normed = [[cm[i][j] / max(cmax, 1) for j in range(k)] for i in range(k)]
    fig, ax = plt.subplots(figsize=(max(5, k), max(4, k)))
    ax.imshow(normed, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (GP label)")
    ax.set_title(title)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix -> {save_path}")


def run_gp_eval(task_dir, existing_classes, f_size, gp_package, rscript_dir):
    """Call check_gp_on_replay.R and return path to predictions CSV."""
    out_csv  = os.path.join(task_dir, "replay_gp_preds.csv")
    r_script = os.path.join(rscript_dir, "check_gp_on_replay.R")
    if not os.path.exists(r_script):
        raise FileNotFoundError(f"check_gp_on_replay.R not found: {r_script}")

    rscript_exe = shutil.which("Rscript") or "Rscript"
    if platform.system() == "Windows":
        rscript_exe = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
    cmd = [
        rscript_exe, r_script,
        "--save_path",        task_dir,
        "--existing_classes", ",".join(str(c) for c in existing_classes),
        "--feature_size",     str(f_size),
        "--GP_package",       gp_package,
        "--out_csv",          out_csv,
    ]
    print(f"  [GP eval] Running check_gp_on_replay.R ...")
    result = subprocess.run(cmd, cwd=rscript_dir, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"check_gp_on_replay.R failed (exit {result.returncode})")
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"Expected GP preds CSV not found: {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate replay point labels using checkpoint models (and optionally GP models)"
    )
    parser.add_argument("--run_dir",    type=str, required=True,
                        help="Path to the run directory (contains config.json)")
    parser.add_argument("--ckpt_task",  type=int, default=None,
                        help="Use this task's checkpoint for ALL evaluations "
                             "(default: same task index as replay dir)")
    parser.add_argument("--tasks",      type=int, nargs="+", default=None,
                        help="Which task dirs to evaluate (default: all task* found)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--plot",       action="store_true",
                        help="Save confusion matrix PNGs per task")
    parser.add_argument("--gp_eval",    action="store_true",
                        help="Also run GP argmax on replay points via check_gp_on_replay.R")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"run_dir not found: {run_dir}")

    cfg        = load_config(run_dir)
    f_size     = cfg["f_size"]
    gp_package = cfg.get("GP_package", "laGP")
    # tasks definition from config (to recover seen_classes_per_task)
    # We reconstruct it the same way CL_Driver.py does.
    tasks_def  = [[0,1,2,3,4],[6],[7],[8],[9]]   # fallback default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rscript_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Device: {device}  |  f_size: {f_size}  |  GP_package: {gp_package}")
    print(f"Run dir: {run_dir}\n")

    if args.tasks is not None:
        task_ids = sorted(args.tasks)
    else:
        task_ids = sorted(
            int(d[4:])
            for d in os.listdir(run_dir)
            if d.startswith("task") and os.path.isdir(os.path.join(run_dir, d))
        )

    if not task_ids:
        sys.exit("No task directories found.")

    # Reconstruct seen classes per task (same logic as CL_Driver.py)
    seen_classes_per_task = [
        sorted(sum(tasks_def[:t+1], []))
        for t in range(len(tasks_def))
    ]

    nn_summary  = []
    gp_summary  = []

    for t in task_ids:
        task_dir  = os.path.join(run_dir, f"task{t}")
        ckpt_task = t if args.ckpt_task is None else args.ckpt_task

        print("=" * 60)
        print(f"Task {t}  |  replay_points from task{t}/  "
              f"|  checkpoint: model_task{ckpt_task}.pt")
        print("=" * 60)

        # --- load replay points ---
        try:
            feats, true_labels = load_replay_points(task_dir, f_size)
        except FileNotFoundError as e:
            print(f"  [skip] {e}\n")
            continue

        classes = sorted(set(true_labels.tolist()))

        # ------------------------------------------------------------------ NN eval
        try:
            model = load_checkpoint(run_dir, ckpt_task, f_size, device=device)
        except FileNotFoundError as e:
            print(f"  [skip NN] {e}")
            model = None

        if model is not None:
            nn_pred  = predict_from_features(model, feats, args.batch_size, device)
            nn_true  = true_labels.tolist()
            nn_preds = nn_pred.tolist()
            nn_correct = sum(1 for t_, p in zip(nn_true, nn_preds) if t_ == p)
            nn_stats   = per_class_stats_list(nn_true, nn_preds, classes)
            nn_wrong_t = [t_ for t_, p in zip(nn_true, nn_preds) if t_ != p]
            nn_wrong_p = [p  for t_, p in zip(nn_true, nn_preds) if t_ != p]

            print()
            print_stats_block("NN", len(nn_true), nn_correct,
                               nn_stats, nn_wrong_t, nn_wrong_p, classes)

            if args.plot:
                cm = confusion_matrix(nn_true, nn_preds, classes)
                plot_confusion(cm, classes,
                               title=f"Task {t} replay | NN model_task{ckpt_task}.pt",
                               save_path=os.path.join(task_dir, f"replay_confusion_nn_ckpt{ckpt_task}.png"))

            nn_summary.append({"task": t, "ckpt": ckpt_task, "n": len(nn_true),
                                "correct": nn_correct, "acc": nn_correct / len(nn_true)})

        # ------------------------------------------------------------------ GP eval
        if args.gp_eval:
            seen = seen_classes_per_task[t] if t < len(seen_classes_per_task) else classes
            print()
            try:
                gp_preds_csv = run_gp_eval(task_dir, seen, f_size, gp_package, rscript_dir)
                gp_true, gp_preds = load_gp_preds_csv(gp_preds_csv)
                gp_correct = sum(1 for t_, p in zip(gp_true, gp_preds) if t_ == p)
                gp_stats   = per_class_stats_list(gp_true, gp_preds, classes)
                gp_wrong_t = [t_ for t_, p in zip(gp_true, gp_preds) if t_ != p]
                gp_wrong_p = [p  for t_, p in zip(gp_true, gp_preds) if t_ != p]

                print_stats_block("GP", len(gp_true), gp_correct,
                                   gp_stats, gp_wrong_t, gp_wrong_p, classes)

                if args.plot:
                    cm = confusion_matrix(gp_true, gp_preds, classes)
                    plot_confusion(cm, classes,
                                   title=f"Task {t} replay | GP argmax",
                                   save_path=os.path.join(task_dir, f"replay_confusion_gp.png"))

                gp_summary.append({"task": t, "n": len(gp_true),
                                    "correct": gp_correct, "acc": gp_correct / len(gp_true)})

            except Exception as e:
                print(f"  [GP eval failed] {e}")

        print()

    # ---------------------------------------------------------------------- summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if nn_summary:
        print(f"\n  NN predictions:")
        print(f"  {'Task':>5}  {'Ckpt':>5}  {'N':>6}  {'Correct':>8}  {'Acc':>7}")
        for r in nn_summary:
            print(f"  {r['task']:>5}  {r['ckpt']:>5}  {r['n']:>6}  "
                  f"{r['correct']:>8}  {100*r['acc']:>6.2f}%")

    if gp_summary:
        print(f"\n  GP argmax predictions:")
        print(f"  {'Task':>5}  {'N':>6}  {'Correct':>8}  {'Acc':>7}")
        for r in gp_summary:
            print(f"  {r['task']:>5}  {r['n']:>6}  {r['correct']:>8}  {100*r['acc']:>6.2f}%")


if __name__ == "__main__":
    main()
