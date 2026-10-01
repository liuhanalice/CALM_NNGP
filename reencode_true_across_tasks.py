"""
reencode_true_across_tasks.py

Companion precompute step for GP_visualize_trajectory.R.

The trajectory plots overlay several layers per task (inducing points,
replay buffer, real test data) that are all encoded by THAT task's own
current model checkpoint -- so if the encoder itself drifts/rescales
between tasks, those layers move together with it. The one static
reference the R script has ("true") is NOT like that: it's the original
class's real TRAINING images, encoded ONCE by the introduction task's
model, then reused as a fixed backdrop in every later panel. That makes
"true" a different coordinate system than the moving layers -- comparing
task 3's inducing points against task-0-encoded "true" conflates "did this
class's region move/shrink" with "did the whole embedding space just get
reorganized by continued training" (the effect discussed at length in the
run's analysis: encoder retraining compresses/relocates feature space even
for real, never-replayed data).

This script computes TWO layers that isolate that effect, both by re-using
the same original real images / points across tasks and only swapping the
encoder snapshot:

1. "orig_now" -- the SAME original real training images (the literal
   originals from the introduction task, not replay-derived), re-encoded
   with EVERY task's own checkpoint (checkpoints/model_task<t>.pt). Any
   movement/shrinkage seen here is purely the shared encoder's own drift,
   nothing to do with replay, decode lossiness, or train/test being
   different image splits.
   Output: <run_root>/GP_original_reencoded_c<class>.csv

2. "prev_inducing_now" -- task (t-1)'s target-only inducing points
   (task<t-1>/Zt_target_c<class>.csv, exported by GP_visualize_trajectory.R
   whenever it runs with --include_inducing), decoded back to images with
   task (t-1)'s OWN decoder, then re-encoded with task t's encoder. Lets
   you see, in task t's own coordinate system, where the GP's actual
   support points from last task now land -- run GP_visualize_trajectory.R
   once first so the Zt_target CSVs exist, then this script, then
   GP_visualize_trajectory.R again to render the full PDF.
   Output: <run_root>/GP_prev_inducing_reencoded_c<class>.csv
   (skipped, with a note, for any task pair missing its Zt_target CSV)

Both outputs share the same column layout: f0..f<feature_size-1>, task,
label -- one row per (point, task) pair, `task` = which checkpoint did the
re-encoding.

Usage:
  python reencode_true_across_tasks.py --run_root runs_mnist_continual/run_20260924_004747 \
      --class 0 --f_size 16
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
from torchvision import datasets, transforms

from NN_AE import CALM_AE_NN


def find_task_dirs(run_root):
    dirs = [d for d in glob.glob(os.path.join(run_root, "task*")) if os.path.isdir(d)]
    nums = [int(re.match(r".*task(\d+)$", d).group(1)) for d in dirs]
    order = np.argsort(nums)
    return [dirs[i] for i in order], [nums[i] for i in order]


def load_model(ckpt_path, f_size, num_classes, device):
    model = CALM_AE_NN(f_size=f_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def write_csv(rows, f_size, out_path):
    f_cols = [f"f{i}" for i in range(f_size)]
    header = f_cols + ["task", "label"]
    with open(out_path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True, help="Run directory containing task0, task1, ... and checkpoints/")
    ap.add_argument("--class", dest="cls", type=int, required=True, help="Class label to re-encode")
    ap.add_argument("--f_size", type=int, default=16, help="Feature dimension (must match training run)")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--n_sample", type=int, default=500, help="Number of original training images to sample for orig_now (same images reused across every task)")
    ap.add_argument("--data_root", type=str, default="./data", help="torchvision MNIST root")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task_dirs, task_nums = find_task_dirs(args.run_root)
    if not task_dirs:
        raise SystemExit(f"No task<N> folders found under {args.run_root}")

    # Introduction task: first task dir whose GPparams_c<cls>.rds exists.
    intro_idx = None
    for i, d in enumerate(task_dirs):
        if os.path.exists(os.path.join(d, f"GPparams_c{args.cls}.rds")):
            intro_idx = i
            break
    if intro_idx is None:
        raise SystemExit(f"Class {args.cls} never appears (no GPparams_c{args.cls}.rds) under {args.run_root}")
    intro_task_num = task_nums[intro_idx]
    print(f"Class {args.cls} introduced at task{intro_task_num}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(args.run_root, "checkpoints")

    # ---- load every available checkpoint from the introduction task onward, once ----
    models = {}
    for t_num in task_nums:
        if t_num < intro_task_num:
            continue
        ckpt_path = os.path.join(ckpt_dir, f"model_task{t_num}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  task{t_num}: no checkpoint at {ckpt_path}, skipping")
            continue
        models[t_num] = load_model(ckpt_path, args.f_size, args.num_classes, device)
    if not models:
        raise SystemExit("No checkpoints found for any task -- nothing to do")

    # =========================================================
    # 1) orig_now: same original real training images, every task's encoder
    # =========================================================
    train_ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transforms.ToTensor())
    idx = [i for i, (_, y) in enumerate(train_ds) if y == args.cls]
    rng = np.random.RandomState(args.seed)
    n_take = min(args.n_sample, len(idx))
    chosen = rng.choice(idx, size=n_take, replace=False)
    real_imgs = torch.stack([train_ds[i][0] for i in chosen]).to(device)  # [N,1,28,28], fixed, real pixels
    print(f"Sampled {real_imgs.shape[0]} original class-{args.cls} training images (of {len(idx)} available)")

    orig_rows = []
    for t_num, model in models.items():
        with torch.no_grad():
            feats = model.extract_adapter_features(real_imgs).cpu().numpy()
        print(f"  orig_now task{t_num}: mean_var={feats.var(axis=0, ddof=1).mean():.4f}")
        for row in feats:
            orig_rows.append(list(row) + [t_num, args.cls])
    write_csv(orig_rows, args.f_size, os.path.join(args.run_root, f"GP_original_reencoded_c{args.cls}.csv"))

    # =========================================================
    # 2) prev_inducing_now: task (t-1)'s target inducing points,
    #    decode(t-1's decoder) -> re-encode(t's encoder)
    # =========================================================
    sorted_t = sorted(models.keys())
    prev_rows = []
    for prev_t, t_num in zip(sorted_t[:-1], sorted_t[1:]):
        zt_path = os.path.join(args.run_root, f"task{prev_t}", f"Zt_target_c{args.cls}.csv")
        if not os.path.exists(zt_path):
            print(f"  prev_inducing_now task{t_num}: missing {zt_path}"
                  f" (run GP_visualize_trajectory.R once first with --include_inducing) -- skipping")
            continue
        zt_arr = np.loadtxt(zt_path, delimiter=",", skiprows=1)
        if zt_arr.ndim == 1:
            zt_arr = zt_arr[None, :]
        Zt = torch.tensor(zt_arr, dtype=torch.float32, device=device)

        with torch.no_grad():
            imgs = torch.clamp(models[prev_t].decode(Zt), 0.0, 1.0)         # decode with task prev_t's decoder
            feats = models[t_num].extract_adapter_features(imgs).cpu().numpy()  # re-encode with task t_num's encoder
        print(f"  prev_inducing_now task{t_num} (from task{prev_t}'s {Zt.shape[0]} inducing points): "
              f"mean_var={feats.var(axis=0, ddof=1).mean():.4f}")
        for row in feats:
            prev_rows.append(list(row) + [t_num, args.cls])

    if prev_rows:
        write_csv(prev_rows, args.f_size, os.path.join(args.run_root, f"GP_prev_inducing_reencoded_c{args.cls}.csv"))
    else:
        print("No Zt_target_c<class>.csv files found for any consecutive task pair -- "
              "skipping GP_prev_inducing_reencoded_c*.csv (run GP_visualize_trajectory.R first)")


if __name__ == "__main__":
    main()
