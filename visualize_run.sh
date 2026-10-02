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
# Before each class's trajectory plot, runs reencode_true_across_tasks.py if
# its outputs (GP_original_reencoded_c<class>.csv,
# GP_prev_inducing_reencoded_c<class>.csv) are missing. That script reads the
# task<t>/Zt_target_c<class>.csv files exported by GP_visualize_trajectory.R,
# so if those are missing too, the order is: trajectory (export Zt_target) ->
# reencode -> trajectory again (final PDF). The reencode step needs the
# python env with torch, <run_root>/checkpoints/model_task<t>.pt and MNIST
# under ./data; if it fails, the trajectory plot is still made without those
# two layers. An existing GP_original_reencoded_c<class>.csv without the
# head_score/head_pred columns (older reencode script) also triggers a rerun.
#
# After each class's trajectory, runs GP_score_orig_now.R (GP and Head scores
# of the original images at every task) if GP_original_reencoded_c<class>.csv
# exists, writing at the run root:
#   <run_root>/GP_score_orig_now_c<class>.pdf
#   <run_root>/GP_score_orig_now_c<class>_summary.csv, _scores.csv
#
# Usage:
#   ./visualize_run.sh runs/run_20260824_122705 [extra Rscript args...]
#
# Any extra arguments are forwarded verbatim to the GP_visualize.R,
# GP_visualize_overlap.R and GP_visualize_trajectory.R calls (not to
# GP_score_orig_now.R), so you can override anything, e.g.:
#   ./visualize_run.sh runs/run_20260824_122705 --n_real=500
#
# Env overrides:
  # N_VIS=1000 N_VIS_OVERLAP=500 N_REAL_TRAJ=500 N_TEST=500 GP_PACKAGE=laGP \
  #   SCORE_THRESHOLD=0.9 TARGET_SCORE_THRESHOLD=0.5 F_SIZE=16 PYTHON=python \
  #   FORCE_REENCODE=1 SKIP_REENCODE=1 \
    # ./visualize_run.sh runs_mnist_continual/run_20260917_180345
#   FORCE_REENCODE=1 reruns reencode_true_across_tasks.py even if its outputs
#   exist; SKIP_REENCODE=1 never runs it. SKIP_SCORE_ORIG=1 skips
#   GP_score_orig_now.R.

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
default_f_size="16"
if [ -f "$config_json" ] && command -v jq >/dev/null 2>&1; then
  cfg_gp_package=$(jq -r '.GP_package // empty' "$config_json")
  cfg_score_threshold=$(jq -r '.GP_score_threshold // empty' "$config_json")
  cfg_f_size=$(jq -r '.f_size // empty' "$config_json")
  [ -n "$cfg_gp_package" ] && default_gp_package="$cfg_gp_package"
  [ -n "$cfg_score_threshold" ] && default_score_threshold="$cfg_score_threshold"
  [ -n "$cfg_f_size" ] && default_f_size="$cfg_f_size"
fi

gp_package="${GP_PACKAGE:-$default_gp_package}"
score_threshold="${SCORE_THRESHOLD:-$default_score_threshold}"
n_vis="${N_VIS:-1000}"
n_vis_overlap="${N_VIS_OVERLAP:-500}"
n_real_traj="${N_REAL_TRAJ:-500}"
n_test_traj="${N_TEST:-500}"
target_score_threshold="${TARGET_SCORE_THRESHOLD:-0.5}"
f_size="${F_SIZE:-$default_f_size}"
python_bin="${PYTHON:-python}"

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
    ${extra_args[@]+"${extra_args[@]}"}

  Rscript GP_visualize_overlap.R -p "$task_dir" \
    --existing_classes "$classes" --GP_package "$gp_package" \
    --score_threshold "$score_threshold" --n_vis "$n_vis_overlap" \
    --data_tr "$train_feat" \
    --out_path "$task_dir" \
    ${extra_args[@]+"${extra_args[@]}"}
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
  run_trajectory() {
    Rscript GP_visualize_trajectory.R -r "$run_root" --class "$1" \
      --feature_size "$f_size" \
      --n_real "$n_real_traj" --n_test "$n_test_traj" \
      --target_score_threshold "$target_score_threshold" \
      --out_path "$run_root" \
      ${extra_args[@]+"${extra_args[@]}"}
  }

  for cls in $all_classes; do
    orig_file="$run_root/GP_original_reencoded_c${cls}.csv"
    prev_file="$run_root/GP_prev_inducing_reencoded_c${cls}.csv"

    need_reencode=0
    if [ "${SKIP_REENCODE:-0}" != "1" ]; then
      if [ "${FORCE_REENCODE:-0}" = "1" ] || [ ! -f "$orig_file" ] || [ ! -f "$prev_file" ]; then
        need_reencode=1
      elif ! head -n 1 "$orig_file" | grep -q "head_score"; then
        echo "$orig_file has no head_score column (older reencode script) -- regenerating"
        need_reencode=1
      fi
    fi

    if [ "$need_reencode" = "1" ]; then
      # reencode reads task<t>/Zt_target_c<cls>.csv, written by the trajectory script
      if ! ls "$run_root"/task*/Zt_target_c"${cls}".csv >/dev/null 2>&1; then
        echo "=== trajectory for class $cls, pass 1 (exporting Zt_target_c${cls}.csv) ==="
        run_trajectory "$cls"
      fi
      echo "=== reencode_true_across_tasks.py for class $cls ==="
      "$python_bin" reencode_true_across_tasks.py --run_root "$run_root" \
        --class "$cls" --f_size "$f_size" \
        || echo "Warning: reencode_true_across_tasks.py failed for class $cls -- plotting without its layers" >&2
    fi

    echo "=== trajectory for class $cls (target_score_threshold: $target_score_threshold) ==="
    run_trajectory "$cls"

    if [ "${SKIP_SCORE_ORIG:-0}" != "1" ]; then
      if [ -f "$orig_file" ]; then
        echo "=== GP/Head scores of original images for class $cls ==="
        Rscript GP_score_orig_now.R -r "$run_root" --class "$cls" \
          --feature_size "$f_size" --GP_package "$gp_package" \
          --score_threshold "$score_threshold" \
          --out_path "$run_root" \
          || echo "Warning: GP_score_orig_now.R failed for class $cls" >&2
      else
        echo "Skipping GP_score_orig_now.R for class $cls (no $orig_file)"
      fi
    fi
  done
fi
