
import pandas as pd
import matplotlib.pyplot as plt


def plotGP_per_class_metrics_over_tasks(
    task_ids,
    metrics=("accuracy", "precision", "recall", "f1"),
    filename="gp_test_metrics.csv",
    title="Per-class metrics over tasks",
    save_path=None
):
    """
    For each task i, reads: {root}/task{i}/{filename}
    Plots subplots for metrics; x=task, y=metric value.
    One line per class with consistent color across all subplots.
    Classes may differ across tasks (missing points are skipped).
    """
    # ---- read all tasks into one df ----
    dfs = []
    for t in task_ids:
        path = f"runs_mnist_continual/run_20260129_010150/task{t}/{filename}" #NOTE: hardcoded root
        df = pd.read_csv(path)
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
    task_ids,
    filename="gp_test_metrics.csv",
    title="Total test accuracy over tasks",
    save_path=None
):
    """
    Reads total accuracy per task from {root}/task{i}/{filename}.
    Assumes column 'total_accuracy' exists and is constant within each CSV.
    Plots x=task, y=total_accuracy.
    """
    xs, ys = [], []
    for t in task_ids:
        path = f"runs_mnist_continual/run_20260129_010150/task{t}/{filename}" #NOTE: hardcoded root
        df = pd.read_csv(path)

        if "total_accuracy" not in df.columns:
            raise ValueError(f"{path} missing 'total_accuracy' column.")

        # same for all rows; take first non-NA
        val = pd.to_numeric(df["total_accuracy"], errors="coerce").dropna()
        if val.empty:
            raise ValueError(f"{path} has no valid numeric total_accuracy.")
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


if __name__ == "__main__":
    # Example usage
    plotGP_per_class_metrics_over_tasks(
        list(range(6)),
        metrics=("accuracy", "precision", "recall", "f1"),
        save_path="GPper_class_metrics.png"
    )

    plotGP_total_accuracy_over_tasks(
        list(range(6)),
        save_path="GP_total_accuracy.png"
    )