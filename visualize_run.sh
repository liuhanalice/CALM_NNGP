#!/bin/bash
# visualize_run.sh
#
# Runs GP_visualize.R and GP_visualize_overlap.R for every task<N> folder
# under a run directory, inferring per-task --existing_classes from the
# GPparams_c*.rds files actually present in each task folder (instead of
# requiring them to be typed out by hand), and pulling --GP_package /
# --score_threshold from the run's config.json when available.
#
# Also runs GP_visualize_trajectory.R once per class that ever appears
# anywhere in the run (not per task -- it internally walks every task from
# that class's introduction onward), writing its outputs at the run root:
#   <run_root>/GP_visualize_trajectory_c<class>.pdf
#   <run_root>/GP_trajectory_spread_c<class>.csv
#
# Usage:
#   ./visualize_run.sh runs/run_20260824_122705 [extra Rscript args...]
#
# Any extra arguments are forwarded verbatim to all three Rscript calls, so
# you can override anything, e.g.:
#   ./visualize_run.sh runs/run_20260824_122705 --n_real=500
#
# Env overrides:
#   N_VIS=1000 N_VIS_OVERLAP=500 N_REAL_TRAJ=500 GP_PACKAGE=laGP \
#     SCORE_THRESHOLD=0.9 TARGET_SCORE_THRESHOLD=0.5 \
#     ./visualize_run.sh runs/run_20260824_122705

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_root, e.g. runs/run_20260824_122705> [extra Rscript args...]" >&2
  exit 1
fi

run_root="${1%/}"
shift
extra_args=("$@")

if [ ! -d "$run_root" ]; then
  echo "Error: run root '$run_root' not found" >&2
  exit 1
fi

if ! command -v Rscript >/dev/null 2>&1; then
  # Best-effort: same env setup as jobs/launch.sh. Harmless no-ops if this
  # isn't that cluster (module/conda absent, or NNGP env not there).
  command -v module >/dev/null 2>&1 && module load R/4.5.1 2>/dev/null
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate NNGP 2>/dev/null
  fi
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Error: Rscript not found on PATH. Load your R environment first, e.g.:" >&2
  echo "  module load R/4.5.1 && conda activate NNGP" >&2
  exit 1
fi

config_json="$run_root/config.json"
default_gp_package="laGP"
default_score_threshold="0.9"
if [ -f "$config_json" ] && command -v jq >/dev/null 2>&1; then
  cfg_gp_package=$(jq -r '.GP_package // empty' "$config_json")
  cfg_score_threshold=$(jq -r '.GP_score_threshold // empty' "$config_json")
  [ -n "$cfg_gp_package" ] && default_gp_package="$cfg_gp_package"
  [ -n "$cfg_score_threshold" ] && default_score_threshold="$cfg_score_threshold"
fi

gp_package="${GP_PACKAGE:-$default_gp_package}"
score_threshold="${SCORE_THRESHOLD:-$default_score_threshold}"
n_vis="${N_VIS:-1000}"
n_vis_overlap="${N_VIS_OVERLAP:-500}"
n_real_traj="${N_REAL_TRAJ:-500}"
target_score_threshold="${TARGET_SCORE_THRESHOLD:-0.5}"

task_dirs=("$run_root"/task*/)
if [ ! -d "${task_dirs[0]}" ]; then
  echo "Error: no task* folders found under '$run_root'" >&2
  exit 1
fi

for task_dir in "${task_dirs[@]}"; do
  task_dir="${task_dir%/}"
  train_feat="$task_dir/train_feat.csv"
  if [ ! -f "$train_feat" ]; then
    echo "Skipping $task_dir (no train_feat.csv)"
    continue
  fi

  classes=$(ls "$task_dir"/GPparams_c*.rds 2>/dev/null \
    | sed -E 's/.*GPparams_c([0-9]+)\.rds/\1/' \
    | sort -n | paste -sd, -)
  if [ -z "$classes" ]; then
    echo "Skipping $task_dir (no GPparams_c*.rds found)"
    continue
  fi

  echo "=== $task_dir (classes: $classes, GP_package: $gp_package, score_threshold: $score_threshold) ==="

  Rscript GP_visualize.R -p "$task_dir" \
    --existing_classes "$classes" --GP_package "$gp_package" \
    --score_threshold "$score_threshold" --n_vis "$n_vis" \
    --data_tr "$train_feat" \
    "${extra_args[@]}"

  Rscript GP_visualize_overlap.R -p "$task_dir" \
    --existing_classes "$classes" --GP_package "$gp_package" \
    --score_threshold "$score_threshold" --n_vis "$n_vis_overlap" \
    --data_tr "$train_feat" \
    --out_path "$task_dir" \
    "${extra_args[@]}"
done

# ---- per-class trajectory: once per class across the WHOLE run, not per task ----
# (GP_visualize_trajectory.R already walks every task from that class's
# introduction onward internally, so calling it once per task would just
# regenerate the same full-run plot redundantly.)
all_classes=$(ls "$run_root"/task*/GPparams_c*.rds 2>/dev/null \
  | sed -E 's/.*GPparams_c([0-9]+)\.rds/\1/' \
  | sort -n -u)

if [ -z "$all_classes" ]; then
  echo "No GPparams_c*.rds found under '$run_root' -- skipping GP_visualize_trajectory.R"
else
  for cls in $all_classes; do
    echo "=== trajectory for class $cls (target_score_threshold: $target_score_threshold) ==="
    Rscript GP_visualize_trajectory.R -r "$run_root" --class "$cls" \
      --n_real "$n_real_traj" --target_score_threshold "$target_score_threshold" \
      --out_path "$run_root" \
      "${extra_args[@]}"
  done
fi
