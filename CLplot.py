import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

import pandas as pd
import ast
import re

_EPOCH_RE = re.compile(r"epoch_(\d+)", re.IGNORECASE)

def read_metrics_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only HEAD stage
    df = df[df["stage"] == "Head"].copy()

    # Extract epoch number from strings like "epoch_1_HEAD";
    def parse_epoch(s):
        s = str(s).strip()
        m = _EPOCH_RE.search(s)
        return int(m.group(1)) if m else None

    df["epoch_idx"] = df["epoch_or_final"].apply(parse_epoch)
    df = df.dropna(subset=["epoch_idx"]).copy()
    df["epoch_idx"] = df["epoch_idx"].astype(int)

    # Sort rows within each task by epoch
    df = df.sort_values(["task_id", "epoch_idx"])

    histories_per_task = []
    epochs_per_task = []

    for task_id, task_df in df.groupby("task_id", sort=True):
        train_acc = task_df["train_acc"].astype(float).tolist()

        # Parse strings like list to list
        test_accs_seen = []
        for v in task_df["test_accs_seen"]:
            if pd.isna(v):
                test_accs_seen.append(None)
            else:
                test_accs_seen.append(ast.literal_eval(str(v)))

        histories_per_task.append({
            "train_acc": train_acc,
            "test_accs_seen": test_accs_seen
        })
        epochs_per_task.append(len(train_acc))

    return histories_per_task, epochs_per_task


import matplotlib.pyplot as plt


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



def plot_z_histograms_by_label(
    csv_path,
    bins=30,
    save_path_dir=None
):
    """
    For each label in train.csv:
        - Create one figure
        - Plot histograms for z0-z9
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Identify z columns automatically
    z_cols = [c for c in df.columns if c.startswith("z")]

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    labels = sorted(df["label"].unique())

    for label in labels:
        df_label = df[df["label"] == label]

        if len(df_label) == 0:
            continue

        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, z in enumerate(z_cols):
            axes[i].hist(df_label[z], bins=bins)
            axes[i].set_title(z)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Count")

        fig.suptitle(f"Latent z Distributions - Label {label}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path_dir is not None:
            save_dir = Path(save_path_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"label_{label}_z_hist.png", dpi=300)
        else:   
            plt.show()


if __name__ == "__main__":
    # root dir
    run_dir="./runs_mnist_continual/run_20260323_140659"

    # Training and Test NN(Head) Accuracy vs. Global epochs
    # csv_path = Path(run_dir) / "metrics.csv"
    # acc_plot_savepath_prefic = Path(run_dir) / "CL"
    # history_dict, epochs_per_task = read_metrics_csv(csv_path)
    # plot_acc_over_all_tasks(history_dict, epochs_per_task, save_path_prefix=acc_plot_savepath_prefic)

    # Logit Histogram
    for i in range(6):
        task_dir = Path(run_dir) / f"task{i}"
        feat_csv_path = task_dir / "train_feat.csv"
        plot_z_histograms_by_label(feat_csv_path, save_path_dir=task_dir)





