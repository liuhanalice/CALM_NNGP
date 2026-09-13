import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

import pandas as pd
import ast
import re

_EPOCH_RE = re.compile(r"epoch_(\d+)", re.IGNORECASE)

def _parse_pyliteral_col(series):
    """Parse a column of Python-literal repr strings (list or dict), e.g.
    "[1.0, 2.0]" or "{0: 91.2, 1: 95.0}" as written by csv.writer. NaN/blank
    entries become None."""
    parsed = []
    for v in series:
        if pd.isna(v):
            parsed.append(None)
        else:
            parsed.append(ast.literal_eval(str(v)))
    return parsed


def _parse_numeric_col(series):
    return pd.to_numeric(pd.Series(list(series)), errors="coerce").tolist()


def read_metrics_csv(csv_path, stage="Head"):
    """
    Read metrics.csv into (histories_per_task, epochs_per_task), keeping
    only rows for the given stage ("Head" or "AE"). Head-stage histories
    carry the accuracy fields plus ce/logit_reg training loss; AE-stage
    histories carry rec/feat_reg training loss. test_ce_mean is logged for
    both stages and included either way.
    """
    df = pd.read_csv(csv_path)
    df = df[df["stage"] == stage].copy()

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

        hist = {
            "train_acc": train_acc,
            "test_accs_seen": _parse_pyliteral_col(task_df.get("test_accs_seen", [])),
            "train_acc_per_class": _parse_pyliteral_col(task_df.get("train_acc_per_class", [])),
            "test_acc_per_class": _parse_pyliteral_col(task_df.get("test_acc_per_class", [])),
            "test_ce_mean": _parse_numeric_col(task_df.get("test_ce_mean", [])),
        }
        if stage == "Head":
            hist["ce"] = _parse_numeric_col(task_df.get("train_ce_head", []))
            hist["logit_reg"] = _parse_numeric_col(task_df.get("train_logit_reg_head", []))
        elif stage == "AE":
            hist["rec"] = _parse_numeric_col(task_df.get("train_rec_ae", []))
            hist["feat_reg"] = _parse_numeric_col(task_df.get("train_feat_reg_ae", []))

        histories_per_task.append(hist)
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



def _global_offsets(epochs_per_task):
    offsets = [0]
    for e in epochs_per_task[:-1]:
        offsets.append(offsets[-1] + e)
    return offsets


def plot_label_wise_accuracy(
    histories_per_task,
    epochs_per_task,
    tasks,
    key,                      # "train_acc_per_class" or "test_acc_per_class"
    title="Label-wise Accuracy over Global Epochs",
    save_path=None
):
    """
    One line per label. A label's line starts at the first global epoch of
    the task that introduces it and continues through the end of training
    (it keeps getting evaluated/trained on via replay in later tasks).
    """
    offsets = _global_offsets(epochs_per_task)
    global_last_epoch = sum(epochs_per_task)

    label_to_task = {lbl: ti for ti, labels in enumerate(tasks) for lbl in labels}

    xs, ys = {}, {}

    for t, hist in enumerate(histories_per_task):
        start = offsets[t]
        for ep_idx, class_acc in enumerate(hist.get(key, []) or []):
            if not class_acc:
                continue
            xg = start + ep_idx + 1
            for lbl, acc in class_acc.items():
                lbl = int(lbl)
                xs.setdefault(lbl, [])
                ys.setdefault(lbl, [])
                xs[lbl].append(xg)
                ys[lbl].append(acc)

    plt.figure(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, lbl in enumerate(sorted(xs.keys())):
        if not xs[lbl]:
            continue
        plt.plot(
            xs[lbl], ys[lbl],
            linewidth=1.8,
            color=cmap(i % 20),
            label=f"Label {lbl} (task {label_to_task.get(lbl, '?')})"
        )

    plt.xlabel("Global Epoch Index")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_task_wise_accuracy(
    histories_per_task,
    epochs_per_task,
    tasks,
    kind="train",   # "train" or "test"
    title="Task-wise Accuracy over Global Epochs",
    save_path=None
):
    """
    One line per task. A task's line starts at the first global epoch where
    it is introduced and continues through the end of training.

    - kind="test" uses the exact per-task test accuracy (history["test_accs_seen"]).
    - kind="train" has no direct per-task training loader, so it is
      approximated as the mean of that task's member labels' per-epoch
      training accuracy (history["train_acc_per_class"]).
    """
    offsets = _global_offsets(epochs_per_task)
    global_last_epoch = sum(epochs_per_task)
    num_tasks = len(tasks)

    xs = [[] for _ in range(num_tasks)]
    ys = [[] for _ in range(num_tasks)]

    if kind == "test":
        for t, hist in enumerate(histories_per_task):
            start = offsets[t]
            for ep_idx, accs_seen in enumerate(hist.get("test_accs_seen", []) or []):
                if accs_seen is None:
                    continue
                xg = start + ep_idx + 1
                for k, acc_k in enumerate(accs_seen):
                    xs[k].append(xg)
                    ys[k].append(acc_k)
    elif kind == "train":
        for t, hist in enumerate(histories_per_task):
            start = offsets[t]
            for ep_idx, class_acc in enumerate(hist.get("train_acc_per_class", []) or []):
                if not class_acc:
                    continue
                xg = start + ep_idx + 1
                for k, labels in enumerate(tasks):
                    accs = [class_acc[lbl] for lbl in labels if lbl in class_acc]
                    if not accs:
                        continue
                    xs[k].append(xg)
                    ys[k].append(float(sum(accs) / len(accs)))
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    plt.figure(figsize=(10, 5))
    for k in range(num_tasks):
        if not xs[k]:
            continue
        plt.plot(xs[k], ys[k], linewidth=2, label=f"Task {k} {tasks[k]}")

    plt.xlabel("Global Epoch Index")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_loss_curves(
    histories_per_task,
    epochs_per_task,
    keys,
    title="Loss over Global Epochs",
    ylabel="Loss",
    save_path=None
):
    """
    One continuous line per history key (e.g. "rec"/"feat_reg" for the AE
    stage, or "ce"/"logit_reg"/"test_ce_mean" for the Head stage), stitched
    across tasks using the same global-epoch-index convention as the
    accuracy plots. Each key is a single scalar per epoch (not broken out
    per task/label), so it's one line for the whole run. None entries are
    skipped rather than plotted as gaps.
    """
    offsets = _global_offsets(epochs_per_task)
    global_last_epoch = sum(epochs_per_task)

    plt.figure(figsize=(10, 5))
    for key in keys:
        xs, ys = [], []
        for t, hist in enumerate(histories_per_task):
            start = offsets[t]
            for ep_idx, v in enumerate(hist.get(key, []) or []):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                xs.append(start + ep_idx + 1)
                ys.append(v)
        if xs:
            plt.plot(xs, ys, linewidth=1.8, label=key)

    plt.xlabel("Global Epoch Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, global_last_epoch + 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
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
    run_dir="./runs_mnist_continual/run_20260324_234303"

    # Training and Test NN(Head) Accuracy vs. Global epochs
    # csv_path = Path(run_dir) / "metrics.csv"
    # acc_plot_savepath_prefic = Path(run_dir) / "CL"
    # history_dict, epochs_per_task = read_metrics_csv(csv_path)
    # plot_acc_over_all_tasks(history_dict, epochs_per_task, save_path_prefix=acc_plot_savepath_prefic)
    #
    # tasks = [[0,1], [2,3], [4,5], [6,7], [8,9]]  # must match the run's task/label layout
    # plot_label_wise_accuracy(history_dict, epochs_per_task, tasks, key="train_acc_per_class",
    #                           save_path=Path(run_dir) / "acc_per_label_train.png")
    # plot_label_wise_accuracy(history_dict, epochs_per_task, tasks, key="test_acc_per_class",
    #                           save_path=Path(run_dir) / "acc_per_label_test.png")
    # plot_task_wise_accuracy(history_dict, epochs_per_task, tasks, kind="train",
    #                          save_path=Path(run_dir) / "acc_per_task_train.png")
    # plot_task_wise_accuracy(history_dict, epochs_per_task, tasks, kind="test",
    #                          save_path=Path(run_dir) / "acc_per_task_test.png")
    #
    # ae_hist, ae_epochs = read_metrics_csv(csv_path, stage="AE")
    # plot_loss_curves(ae_hist, ae_epochs, keys=["rec", "feat_reg"],
    #                   title="AE Stage-1 Loss vs Global Epochs",
    #                   save_path=Path(run_dir) / "loss_ae_stage1.png")
    # plot_loss_curves(history_dict, epochs_per_task, keys=["ce", "logit_reg", "test_ce_mean"],
    #                   title="Head Stage-2 Loss vs Global Epochs",
    #                   save_path=Path(run_dir) / "loss_head_stage2.png")

    # Logit Histogram
    for i in range(6):
        task_dir = Path(run_dir) / f"task{i}"
        feat_csv_path = task_dir / "train_feat.csv"
        plot_z_histograms_by_label(feat_csv_path, save_path_dir=task_dir)





