#!/bin/bash
# visualize_run.sh
#
# Runs GP_visualize.R and GP_visualize_overlap.R for every task<N> folder
# under a run directory, inferring per-task --existing_classes from the
# GPparams_c*.rds files actually present in each task folder (instead of
# requiring them to be typed out by hand), and pulling --GP_package /
# --score_threshold from the run's config.json when available.
#
# Usage:
#   ./visualize_run.sh runs/run_20260824_122705 [extra Rscript args...]
#
# Any extra arguments are forwarded verbatim to both Rscript calls, so you
# can override anything, e.g.:
#   ./visualize_run.sh runs/run_20260824_122705 --n_real=500
#
# Env overrides:
#   N_VIS=1000 N_VIS_OVERLAP=500 GP_PACKAGE=laGP SCORE_THRESHOLD=0.9 \
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
