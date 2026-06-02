"""
Compare performance across multiple continual learning runs.

Usage:
    python compare_runs.py run_20260529_085555 run_20260529_094512 run_20260529_221443 run_20260529_221612 --save comparison.png
    python compare_runs.py --runs_root /path/to/runs run1 run2 run3
    python compare_runs.py run1 run2 --save comparison.png
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def diff_configs(configs: dict[str, dict]) -> dict[str, dict]:
    """Return only keys that differ across runs."""
    all_keys = set().union(*[c.keys() for c in configs.values()])
    differing = {}
    for key in sorted(all_keys):
        values = {run: c.get(key, "<missing>") for run, c in configs.items()}
        if len(set(str(v) for v in values.values())) > 1:
            differing[key] = values
    return differing


def load_head_final_metrics(run_dir):
    """Per-task final head test accuracy from metrics.csv."""
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    finals = df[df["stage"] == "Final"][["task_id", "test_acc_mean"]].copy()
    finals = finals.sort_values("task_id").set_index("task_id")
    finals.index = [f"task{i}" for i in finals.index]
    return finals["test_acc_mean"]


def load_gp_total_accuracy(run_dir):
    """Per-task GP total_accuracy (last row of each task's gp_test_metrics.csv)."""
    results = {}
    task_dirs = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("task")],
        key=lambda x: int(x.replace("task", "")),
    )
    for task in task_dirs:
        gp_file = os.path.join(run_dir, task, "gp_test_metrics.csv")
        if os.path.exists(gp_file):
            df = pd.read_csv(gp_file)
            # total_accuracy is the same for all rows in a file — just take first
            results[task] = df["total_accuracy"].iloc[0]
        else:
            results[task] = float("nan")
    return pd.Series(results)


def load_gp_per_class_f1(run_dir):
    """Per-task mean F1 across classes from gp_test_metrics.csv."""
    results = {}
    task_dirs = sorted(
        [d for d in os.listdir(run_dir) if d.startswith("task")],
        key=lambda x: int(x.replace("task", "")),
    )
    for task in task_dirs:
        gp_file = os.path.join(run_dir, task, "gp_test_metrics.csv")
        if os.path.exists(gp_file):
            df = pd.read_csv(gp_file)
            results[task] = df["f1"].mean()
        else:
            results[task] = float("nan")
    return pd.Series(results)


def make_label(run_name, diff, multiline=False):
    """Short label: run name + differing config values."""
    if not diff:
        return run_name
    extras = ", ".join(f"{k}={v[run_name]}" for k, v in diff.items())
    sep = "\n" if multiline else " "
    return f"{run_name}{sep}({extras})"


def plot_comparison(data: dict[str, pd.Series], title: str, ylabel: str, ax):
    tasks = None
    for label, series in data.items():
        if tasks is None:
            tasks = list(series.index)
        ax.plot(tasks, series.values, marker="o", label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)


def print_table(title, df):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    parser = argparse.ArgumentParser(description="Compare continual learning runs.")
    parser.add_argument("runs", nargs="+", help="Run folder names to compare")
    parser.add_argument(
        "--runs_root",
        default="./runs_mnist_continual",
        help="Root directory containing run folders (default: ./runs_mnist_continual)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["head_acc", "gp_total_acc", "gp_f1", "all"],
        default=["all"],
        help="Metrics to display (default: all)",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save plot to this path instead of showing interactively (e.g. comparison.png)",
    )
    args = parser.parse_args()

    show_all = "all" in args.metrics
    show_head = show_all or "head_acc" in args.metrics
    show_gp_total = show_all or "gp_total_acc" in args.metrics
    show_gp_f1 = show_all or "gp_f1" in args.metrics

    run_dirs = {}
    for run_name in args.runs:
        path = os.path.join(args.runs_root, run_name)
        if not os.path.isdir(path):
            print(f"[ERROR] Run directory not found: {path}", file=sys.stderr)
            sys.exit(1)
        run_dirs[run_name] = path

    # ── Config diff ──────────────────────────────────────────────────────────
    configs = {name: load_config(d) for name, d in run_dirs.items()}
    diff = diff_configs(configs)

    print(f"\n{'='*60}")
    print("  Config differences")
    print(f"{'='*60}")
    if not diff:
        print("  (all configs identical)")
    else:
        max_key_len = max(len(k) for k in diff)
        header = f"  {'param':<{max_key_len}}  " + "  ".join(
            f"{r:<20}" for r in run_dirs
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key, values in diff.items():
            row = f"  {key:<{max_key_len}}  " + "  ".join(
                f"{str(values[r]):<20}" for r in run_dirs
            )
            print(row)

    # ── Per-task metrics ─────────────────────────────────────────────────────
    labels = {name: make_label(name, diff, multiline=False) for name in run_dirs}
    plot_labels = {name: make_label(name, diff, multiline=True) for name in run_dirs}

    head_df = gp_df = f1_df = None

    if show_head:
        head_df = pd.DataFrame(
            {labels[name]: load_head_final_metrics(d) for name, d in run_dirs.items()}
        )
        print_table("Head test accuracy (per task after training that task)", head_df)
        avg = head_df.mean()
        print(f"\n  Average:  {avg.to_string(float_format=lambda x: f'{x:.4f}')}")

    if show_gp_total:
        gp_df = pd.DataFrame(
            {labels[name]: load_gp_total_accuracy(d) for name, d in run_dirs.items()}
        )
        print_table("GP total accuracy (per task)", gp_df)
        avg = gp_df.mean()
        print(f"\n  Average:  {avg.to_string(float_format=lambda x: f'{x:.4f}')}")

    if show_gp_f1:
        f1_df = pd.DataFrame(
            {labels[name]: load_gp_per_class_f1(d) for name, d in run_dirs.items()}
        )
        print_table("GP mean per-class F1 (per task)", f1_df)
        avg = f1_df.mean()
        print(f"\n  Average:  {avg.to_string(float_format=lambda x: f'{x:.4f}')}")

    # ── Summary row ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Summary (averages across all tasks)")
    print(f"{'='*60}")
    summary_rows = {}
    if show_head:
        summary_rows["head_acc (mean)"] = {
            labels[name]: load_head_final_metrics(d).mean() for name, d in run_dirs.items()
        }
    if show_gp_total:
        summary_rows["gp_total_acc (mean)"] = {
            labels[name]: load_gp_total_accuracy(d).mean() for name, d in run_dirs.items()
        }
    if show_gp_f1:
        summary_rows["gp_f1 (mean)"] = {
            labels[name]: load_gp_per_class_f1(d).mean() for name, d in run_dirs.items()
        }

    summary_df = pd.DataFrame(summary_rows).T
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    active = [(head_df, "Head test accuracy", "Accuracy"),
              (gp_df,   "GP total accuracy",  "Accuracy"),
              (f1_df,   "GP mean per-class F1", "F1")]
    active = [(df, t, y) for df, t, y in active if df is not None]

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    config_str = "  |  ".join(
        f"{k}: " + " vs ".join(str(v) for v in vals.values())
        for k, vals in diff.items()
    ) or "identical configs"
    fig.suptitle(f"Run Comparison", fontsize=10, y=0.96)

    rename = {labels[n]: plot_labels[n] for n in run_dirs}
    for ax, (df, title, ylabel) in zip(axes[0], active):
        plot_comparison({rename[col]: df[col] for col in df.columns}, title, ylabel, ax)

    # Summary bar chart appended as the last subplot
    if summary_rows:
        short_labels = {name: f"Run {i+1}" for i, name in enumerate(run_dirs)}
        col_to_short = {labels[name]: short_labels[name] for name in run_dirs}

        n_runs = len(run_dirs)
        fig_height = 4 + 0.25 * n_runs
        text_frac = min(0.35, 0.25 * n_runs / fig_height)

        fig2, ax2 = plt.subplots(figsize=(max(4, len(summary_rows) * 1.5), fig_height))
        summary_df.rename(columns=col_to_short).T.plot(kind="bar", ax=ax2, rot=0)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Mean accuracy / F1 across all tasks")
        ax2.set_ylabel("Score")
        ax2.legend(fontsize=7)
        ax2.grid(True, axis="y", alpha=0.3)

        legend_lines = [f"Run {i+1}: {labels[name]}" for i, name in enumerate(run_dirs)]
        fig2.text(0.01, 0.005, "\n".join(legend_lines), fontsize=7, va="bottom",
                  family="monospace")
        fig2.tight_layout(rect=[0, text_frac, 1, 1])

    fig.tight_layout()

    if args.save:
        base, ext = os.path.splitext(args.save)
        fig.savefig(args.save, bbox_inches="tight", dpi=150)
        if summary_rows:
            fig2.savefig(f"{base}_summary{ext}", bbox_inches="tight", dpi=150)
        print(f"Saved plots to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
