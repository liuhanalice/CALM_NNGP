# GP_visualize_trajectory.R
#
# For ONE class, traces how the points feeding continual-learning replay for
# that class move/shrink across tasks, relative to the class's true training
# data at the task it was first introduced.
#
# Why this can drift: for an old class, all_data_X used to refit its GP at
# task t is read from task<t>/train_feat.csv (GP_train.R, ~line 132), which
# for that class is NOT raw data -- it's re-encoded features of images that
# were decode()'d from task<t-1>'s GP-sampled replay_points.csv and folded
# into task t's training set (CL_Driver.py, ~lines 1335-1362). Each task's
# replay buffer only keeps candidates scoring >= --GP_score_threshold under
# that class's own GP (GP_sample.R). So with --retrain_all_gp, each task's
# "training data" for an old class is itself the prior task's
# high-confidence subset (decode/encode round-tripped), and inducing points
# (GPparams_c<label>.rds$Z_t) are a random subset of THAT. If each
# accept/refit cycle keeps a strict subset of the support, this compounds
# task over task -- this script makes that visible/measurable.
#
# Caveat: GPparams_c<label>.rds$Z_t is NOT purely this class's inducing
# points. train_GP_v3 / train_GP_laGP (utils.R) append X_otc -- a handful of
# OTHER classes' points -- so the reconstructed GP also learns Y~=0 off the
# decision boundary instead of extrapolating to prior~=1 OOD. Those rows
# would otherwise corrupt this class's trajectory (wrong centroid, inflated
# variance, UMAP pulled toward other classes). For laGP runs (the only
# package that persists Y_Z_t), we filter Z_t down to rows whose pseudo-target
# Y_Z_t >= --target_score_threshold, i.e. the target-class rows only; gplite
# runs don't persist Y_Z_t and can't be filtered here, so all of Z_t is kept.
#
# Also overlays each task's REAL, held-out test-set data for this class
# (task<t>/test_feat.csv, re-encoded by THAT task's current model -- see
# export_task_csvs in CL_Driver.py, list_test_loader=seen_test_loaders is
# cumulative, so every seen class's real test points get re-extracted every
# task). This is the one thing in the whole pipeline that's never replayed,
# decoded, or re-encoded from a previous task's output -- it's the fixed
# ground truth the training-side data (inducing/replay) should still cover.
# The per-task test DataLoader is a fixed, shuffle=False Subset built once
# and reused every task (make_task_loaders), so a given class's row order in
# test_feat.csv is identical task to task -- we exploit that by sampling the
# SAME row indices every task, so "test" tracks the exact same underlying
# images across tasks (paired, like orig_now), not just a fresh same-
# population draw each time.
# Comparing "test" against "true" (the fixed intro-task baseline) shows
# whether the ENCODER itself is drifting/shrinking for real data, independent
# of anything replay-specific; comparing "test" against "inducing"/"replay"
# within the same task shows whether the training-side data still covers
# what real data for this class actually looks like right now.
# (--include_test is now off by default: test turned out to track orig_now
# closely, so pages 1-4 show train_now -- task t's actual train_feat.csv rows
# for this class, the pool the inducing points are sampled from -- instead.)
#
# "true" itself, though, is a STATIC reference: it's the original real
# training images encoded ONCE, by the introduction task's model, then
# reused as a fixed backdrop in every later panel -- while inducing/replay/
# test are all re-encoded by THAT task's own current model. So "true" isn't
# in the same coordinate system as later tasks' layers; comparing them
# conflates "did class `cls` move/shrink" with "did the whole embedding
# space get reorganized by continued training" (see run analysis: the
# shared encoder compresses/relocates feature space for real data too, not
# just replay). optionally overlays a 4th layer, "orig_now": the SAME
# original real training images as "true", but re-encoded by EACH task's
# own checkpoint (task i uses model_task<i>.pt -- see
# reencode_true_across_tasks.py, which must be run first to produce
# <run_root>/GP_original_reencoded_c<class>.csv). Same images throughout,
# only the encoder snapshot changes, so any movement/shrinkage in "orig_now"
# is purely the shared encoder's own drift -- nothing to do with replay,
# decode lossiness, or train/test being different image splits.
#
# Two more optional layers feed a dedicated overlap-check page (see Pages
# 8/9): "train_now" is task t's ACTUAL train_feat.csv rows for this class --
# what GP_train.R really trained on (already correctly encoded by task t's
# own model, by construction). "prev_inducing_now" is task (t-1)'s
# target-only inducing points, decoded with task (t-1)'s decoder and
# re-encoded with task t's encoder (reencode_true_across_tasks.py again,
# reading task<t-1>/Zt_target_c<class>.csv -- exported by THIS script
# whenever --include_inducing runs, so run this script once first, then the
# python step, then this script again to pick both new CSVs up). Together
# with "orig_now", all three are on the SAME task-t encoder, so overlap (or
# lack of it) between them is a direct, apples-to-apples read of whether
# this task's actual training data still covers the GP's own last-task
# support and the original real cluster.
#
# Three views:
#   1. VISUAL (UMAP): one shared, UNSUPERVISED UMAP embedding (fit jointly
#      over the true introduction-task data + every later task's inducing
#      points, replay buffer, and real test data for this class) -- an
#      overlay with a centroid path per task, and a small-multiples facet by
#      task. Good for an at-a-glance "do these clouds separate" read, but
#      UMAP distorts distances/density, so it is NOT reliable for judging
#      whether a cloud has shifted vs. shrunk -- use the PCA view for that.
#   2. VISUAL (PCA): the same layers, but projected with a shared *linear*
#      PCA (fit once, jointly, like the UMAP) instead. Unlike UMAP, a linear
#      projection preserves relative distances/spread along the axes it
#      keeps, so a cloud moving vs. a cloud shrinking actually look different
#      here -- this is the view to use for the shift-vs-shrink question.
#   3. QUANTITATIVE: mean per-dimension variance and mean distance to the
#      true data's centroid, computed directly in raw feature space (not
#      UMAP/PCA-distorted), plotted over task index against the true data's
#      own values as a dashed baseline.
#
# Output: <out_path>/GP_visualize_trajectory_c<class>.pdf
#         <out_path>/GP_trajectory_spread_c<class>.csv
#
# Usage:
#   Rscript GP_visualize_trajectory.R -r runs_mnist_continual/run_20260917_180345 --class 0

library(optparse)
library(grid)
source("utils.R")   # loads ggplot2, umap, dplyr, stats, etc.

option_list <- list(
  make_option(c("-r", "--run_root"),       type = "character", default = NULL,
              help = "Run directory containing task0, task1, ... subfolders"),
  make_option(c("--class"),                type = "numeric",   default = NULL,
              help = "Class label to trace across tasks"),
  make_option(c("-f", "--feature_size"),   type = "numeric",   default = 16,
              help = "Feature dimension (must match GP_train.R) [default: %default]"),
  make_option(c("--n_real"),               type = "numeric",   default = 500,
              help = "True training points (at introduction task) to plot as baseline [default: %default]"),
  make_option(c("--include_inducing"),     type = "logical",   default = TRUE,
              help = "Overlay each task's GP inducing points (GPparams_c<class>.rds$Z_t) [default: %default]"),
  make_option(c("--target_score_threshold"), type = "numeric", default = 0.5,
              help = "Min Y_Z_t pseudo-target to keep a Z_t row as target-class (laGP only; excludes X_otc other-class anchor rows) [default: %default]"),
  make_option(c("--include_replay"),       type = "logical",   default = TRUE,
              help = "Overlay each task's GP-sampled replay buffer (replay_points.csv) [default: %default]"),
  make_option(c("--include_test"),         type = "logical",   default = FALSE,
              help = "Overlay each task's REAL held-out test data for this class (test_feat.csv, never replayed); off by default since it tracks orig_now closely -- train_now is shown in its place [default: %default]"),
  make_option(c("--n_test"),               type = "numeric",   default = 500,
              help = "Real test points per task to sample for plotting/stats [default: %default]"),
  make_option(c("--include_orig_reencoded"), type = "logical", default = TRUE,
              help = "Overlay the SAME original real training images re-encoded by EACH task's own checkpoint (reads <run_root>/GP_original_reencoded_c<class>.csv produced by reencode_true_across_tasks.py; silently skipped if missing) [default: %default]"),
  make_option(c("--include_train_now"),    type = "logical",   default = TRUE,
              help = "Overlay each task's ACTUAL train_feat.csv rows for this class -- what GP_train.R really trained on that task (real data at introduction, decode/re-encode round-tripped replay afterward) [default: %default]"),
  make_option(c("--n_train_now"),          type = "numeric",   default = 500,
              help = "Current-training-data points per task to sample for plotting/stats [default: %default]"),
  make_option(c("--include_prev_inducing_now"), type = "logical", default = TRUE,
              help = "Overlay the PREVIOUS task's target inducing points, decoded with that task's decoder and re-encoded with THIS task's encoder (reads <run_root>/GP_prev_inducing_reencoded_c<class>.csv produced by reencode_true_across_tasks.py; silently skipped if missing) [default: %default]"),
  make_option(c("--out_path"),             type = "character", default = NULL,
              help = "Directory to write outputs [default: --run_root]"),
  make_option(c("--seed"),                 type = "numeric",   default = 42,
              help = "Random seed [default: %default]")
)

parser <- OptionParser(option_list = option_list)
args   <- parse_args(parser)

if (is.null(args$run_root)) stop("--run_root is required")
if (is.null(args$class))    stop("--class is required")
set.seed(args$seed)

out_path <- if (is.null(args$out_path)) args$run_root else args$out_path
prepare_save_dir(out_path)

f      <- as.integer(args$feature_size)
cls    <- args$class
n_real <- as.integer(args$n_real)
n_test <- as.integer(args$n_test)
n_train_now <- as.integer(args$n_train_now)

# GP_train_size_per_class (n_tr in GP_train.R / load_data_per_class, utils.R:51-66):
# if a class's train_feat.csv pool ever exceeds this, GP_train.R itself draws a
# random n_tr-sized X_t from it via ITS OWN RNG stream -- which we have no way
# to reconstruct here, so "train_now" (below) would then be an independent
# resample of the pool, not guaranteed to be the exact rows GP_train.R trained
# on (still the same distribution, just not point-for-point identical). Read
# it from config.json, if present, so we can at least flag when that's the
# case rather than silently assume train_now == a subset of the real X_t.
gp_train_size_per_class <- NA_integer_
config_json_file <- paste0(args$run_root, "/config.json")
if (file.exists(config_json_file)) {
  cfg_lines <- readLines(config_json_file, warn = FALSE)
  m <- regmatches(cfg_lines, regexpr('"GP_train_size_per_class"[[:space:]]*:[[:space:]]*[0-9]+', cfg_lines))
  m <- m[nzchar(m)]
  if (length(m) > 0) {
    gp_train_size_per_class <- as.integer(sub('.*:[[:space:]]*', '', m[1]))
  }
}

# ---- find task dirs, sorted numerically ----
task_dirs_all <- list.dirs(args$run_root, recursive = FALSE)
task_dirs_all <- task_dirs_all[grepl("/task[0-9]+$", task_dirs_all)]
if (length(task_dirs_all) == 0) stop(paste("No task<N> folders found under", args$run_root))
task_nums_all <- as.integer(sub(".*/task([0-9]+)$", "\\1", task_dirs_all))
ord <- order(task_nums_all)
task_dirs_all <- task_dirs_all[ord]
task_nums_all <- task_nums_all[ord]

# ---- introduction task: first task dir containing GPparams_c<class>.rds ----
has_params <- file.exists(paste0(task_dirs_all, "/GPparams_c", cls, ".rds"))
if (!any(has_params)) {
  stop(paste0("Class ", cls, " never appears (no GPparams_c", cls, ".rds under ", args$run_root, ")"))
}
intro_idx      <- min(which(has_params))
intro_task_dir <- task_dirs_all[intro_idx]
intro_task_num <- task_nums_all[intro_idx]
print(paste0("Class ", cls, " introduced at task", intro_task_num, " (", intro_task_dir, ")"))

# ---- true training data at introduction task ----
true_csv <- paste0(intro_task_dir, "/train_feat.csv")
if (!file.exists(true_csv)) stop(paste("Missing", true_csv))
true_df <- read.csv(true_csv)
true_df$label <- as.numeric(as.character(true_df$label))
true_rows <- true_df[true_df$label == cls, 1:f, drop = FALSE]
if (nrow(true_rows) == 0) stop(paste0("No rows with label==", cls, " in ", true_csv))
n_take <- min(n_real, nrow(true_rows))
true_X <- as.matrix(true_rows[sample(nrow(true_rows), n_take), , drop = FALSE])
print(paste0("True training data at introduction: ", nrow(true_X),
             " points (of ", nrow(true_rows), " available) from task", intro_task_num))

# ---- optional: same original real training images, re-encoded per task ----
orig_reencoded_df <- NULL
if (isTRUE(args$include_orig_reencoded)) {
  orig_reencoded_file <- paste0(args$run_root, "/GP_original_reencoded_c", cls, ".csv")
  if (file.exists(orig_reencoded_file)) {
    orig_reencoded_df <- read.csv(orig_reencoded_file)
    orig_reencoded_df$label <- as.numeric(as.character(orig_reencoded_df$label))
  } else {
    print(paste0("Note: --include_orig_reencoded is on but ", orig_reencoded_file,
                  " doesn't exist yet. Generate it with: python reencode_true_across_tasks.py --run_root ",
                  args$run_root, " --class ", cls, " --f_size ", f))
  }
}

# ---- optional: previous task's target inducing points, decoded with that
# task's decoder and re-encoded with THIS task's encoder (needs a prior run
# of this script to have exported task<t>/Zt_target_c<class>.csv, which the
# inducing block below does whenever --include_inducing is on) ----
prev_inducing_df <- NULL
if (isTRUE(args$include_prev_inducing_now)) {
  prev_inducing_file <- paste0(args$run_root, "/GP_prev_inducing_reencoded_c", cls, ".csv")
  if (file.exists(prev_inducing_file)) {
    prev_inducing_df <- read.csv(prev_inducing_file)
    prev_inducing_df$label <- as.numeric(as.character(prev_inducing_df$label))
  } else {
    print(paste0("Note: --include_prev_inducing_now is on but ", prev_inducing_file,
                  " doesn't exist yet. Generate it with: python reencode_true_across_tasks.py --run_root ",
                  args$run_root, " --class ", cls, " --f_size ", f,
                  " (run this script once first so task<t>/Zt_target_c", cls, ".csv exists for it to read)"))
  }
}

# ---- collect inducing / replay points from every task from introduction onward ----
combined <- true_X
type_col <- rep("true", nrow(true_X))
task_col <- rep(intro_task_num, nrow(true_X))

# Row order within a class's slice of test_feat.csv is identical every task
# (CL_Driver.py's per-task test DataLoader is a fixed, shuffle=False Subset,
# built once and reused/re-evaluated every task -- see make_task_loaders /
# seen_test_loaders). So picking the SAME row indices every task tracks the
# SAME underlying real images across tasks -- like orig_now, not just an
# independent same-population draw each time. Sampled once, on first use.
test_sample_idx <- NULL

true_center <- colMeans(true_X)
spread_rows <- list(data.frame(
  task = intro_task_num, type = "true", n = nrow(true_X),
  mean_var = mean(diag(cov(true_X))),
  mean_dist_to_true_center = mean(sqrt(rowSums(sweep(true_X, 2, true_center)^2)))
))

later_idx <- which(task_nums_all >= intro_task_num)
for (i in later_idx) {
  t_dir <- task_dirs_all[i]
  t_num <- task_nums_all[i]

  if (isTRUE(args$include_inducing)) {
    params_file <- paste0(t_dir, "/GPparams_c", cls, ".rds")
    if (file.exists(params_file)) {
      p_gp   <- readRDS(params_file)
      Zt_all <- as.matrix(p_gp$Z_t)
      if (!is.null(p_gp$Y_Z_t)) {
        # Z_t = rbind(target-class rows, X_otc other-class anchor rows);
        # Y_Z_t carries the matching pseudo-targets (~1 target, ~0 other) --
        # see train_GP_laGP in utils.R. Keep only the target-class rows so
        # the other classes' anchor points don't distort this class's
        # trajectory/spread.
        is_target <- as.numeric(p_gp$Y_Z_t) >= args$target_score_threshold
        Zt <- Zt_all[is_target, , drop = FALSE]
      } else {
        # gplite doesn't persist Y_Z_t, so target vs. other-class rows can't
        # be told apart here -- fall back to all of Z_t (may include X_otc).
        Zt <- Zt_all
      }
      if (nrow(Zt) > 1) {
        combined <- rbind(combined, Zt)
        type_col <- c(type_col, rep("inducing", nrow(Zt)))
        task_col <- c(task_col, rep(t_num, nrow(Zt)))
        spread_rows[[length(spread_rows) + 1]] <- data.frame(
          task = t_num, type = "inducing", n = nrow(Zt),
          mean_var = mean(diag(cov(Zt))),
          mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Zt, 2, true_center)^2)))
        )
        # Export target-only inducing points as plain feature CSV -- input
        # for reencode_true_across_tasks.py's decode(this task)->re-encode
        # (next task) step, which produces prev_inducing_now.
        Zt_out <- as.data.frame(Zt)
        colnames(Zt_out) <- paste0("f", seq_len(f) - 1)
        write.csv(Zt_out, file = paste0(t_dir, "/Zt_target_c", cls, ".csv"), row.names = FALSE)
      }
    }
  }

  if (isTRUE(args$include_replay)) {
    replay_file <- paste0(t_dir, "/replay_points.csv")
    if (file.exists(replay_file)) {
      rdf <- read.csv(replay_file)
      rdf$label <- as.numeric(as.character(rdf$label))
      Xr <- as.matrix(rdf[rdf$label == cls, 1:f, drop = FALSE])
      if (nrow(Xr) > 1) {
        combined <- rbind(combined, Xr)
        type_col <- c(type_col, rep("replay", nrow(Xr)))
        task_col <- c(task_col, rep(t_num, nrow(Xr)))
        spread_rows[[length(spread_rows) + 1]] <- data.frame(
          task = t_num, type = "replay", n = nrow(Xr),
          mean_var = mean(diag(cov(Xr))),
          mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Xr, 2, true_center)^2)))
        )
      }
    }
  }

  if (isTRUE(args$include_test)) {
    # REAL, held-out test data for this class, re-encoded by task t's own
    # model (export_task_csvs in CL_Driver.py -- list_test_loader is
    # cumulative, so class `cls` keeps appearing here every task once
    # introduced). Never replayed/decoded/re-encoded from a prior task's
    # output -- this is the fixed ground truth everything else should track.
    test_file <- paste0(t_dir, "/test_feat.csv")
    if (file.exists(test_file)) {
      tdf <- read.csv(test_file)
      tdf$label <- as.numeric(as.character(tdf$label))
      test_rows <- tdf[tdf$label == cls, 1:f, drop = FALSE]
      if (nrow(test_rows) > 1) {
        if (is.null(test_sample_idx)) {
          n_take_test <- min(n_test, nrow(test_rows))
          test_sample_idx <- sample(nrow(test_rows), n_take_test)
        }
        idx_use <- test_sample_idx[test_sample_idx <= nrow(test_rows)]
        Xtest <- as.matrix(test_rows[idx_use, , drop = FALSE])
        combined <- rbind(combined, Xtest)
        type_col <- c(type_col, rep("test", nrow(Xtest)))
        task_col <- c(task_col, rep(t_num, nrow(Xtest)))
        spread_rows[[length(spread_rows) + 1]] <- data.frame(
          task = t_num, type = "test", n = nrow(Xtest),
          mean_var = mean(diag(cov(Xtest))),
          mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Xtest, 2, true_center)^2)))
        )
      }
    }
  }

  if (!is.null(orig_reencoded_df)) {
    # Same original real training images as "true", re-encoded by task
    # t_num's own checkpoint (reencode_true_across_tasks.py). Isolates pure
    # encoder drift: same images every task, only the encoder snapshot
    # changes.
    orig_rows <- orig_reencoded_df[orig_reencoded_df$task == t_num & orig_reencoded_df$label == cls, 1:f, drop = FALSE]
    if (nrow(orig_rows) > 1) {
      Xorig <- as.matrix(orig_rows)
      combined <- rbind(combined, Xorig)
      type_col <- c(type_col, rep("orig_now", nrow(Xorig)))
      task_col <- c(task_col, rep(t_num, nrow(Xorig)))
      spread_rows[[length(spread_rows) + 1]] <- data.frame(
        task = t_num, type = "orig_now", n = nrow(Xorig),
        mean_var = mean(diag(cov(Xorig))),
        mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Xorig, 2, true_center)^2)))
      )
    }
  }

  if (isTRUE(args$include_train_now)) {
    # What GP_train.R ACTUALLY trained on for this class at task t_num --
    # real data at the introduction task, decode/re-encode round-tripped
    # replay from the prior task afterward (see file header). Already
    # correctly encoded by task t_num's own model by construction (train_feat.csv
    # is written by export_task_csvs using that task's current model).
    train_now_file <- paste0(t_dir, "/train_feat.csv")
    if (file.exists(train_now_file)) {
      trdf <- read.csv(train_now_file)
      trdf$label <- as.numeric(as.character(trdf$label))
      train_now_rows <- trdf[trdf$label == cls, 1:f, drop = FALSE]
      if (!is.na(gp_train_size_per_class) && nrow(train_now_rows) > gp_train_size_per_class) {
        print(paste0("Warning: task", t_num, " has ", nrow(train_now_rows),
                      " train_feat.csv rows for class ", cls, " (> GP_train_size_per_class=",
                      gp_train_size_per_class, "), so GP_train.R itself subsampled X_t from this pool -- ",
                      "train_now here is an independent resample of the SAME pool, not guaranteed to be ",
                      "the exact rows GP_train.R actually trained on"))
      }
      if (nrow(train_now_rows) > 1) {
        n_take_train_now <- min(n_train_now, nrow(train_now_rows))
        Xtrain_now <- as.matrix(train_now_rows[sample(nrow(train_now_rows), n_take_train_now), , drop = FALSE])
        combined <- rbind(combined, Xtrain_now)
        type_col <- c(type_col, rep("train_now", nrow(Xtrain_now)))
        task_col <- c(task_col, rep(t_num, nrow(Xtrain_now)))
        spread_rows[[length(spread_rows) + 1]] <- data.frame(
          task = t_num, type = "train_now", n = nrow(Xtrain_now),
          mean_var = mean(diag(cov(Xtrain_now))),
          mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Xtrain_now, 2, true_center)^2)))
        )
      }
    }
  }

  if (!is.null(prev_inducing_df)) {
    # Previous task's target inducing points, decode(prev task's decoder) ->
    # re-encode(THIS task's encoder) -- see reencode_true_across_tasks.py.
    # No data for the introduction task itself (no prior task's GP exists).
    prev_rows <- prev_inducing_df[prev_inducing_df$task == t_num & prev_inducing_df$label == cls, 1:f, drop = FALSE]
    if (nrow(prev_rows) > 1) {
      Xprev <- as.matrix(prev_rows)
      combined <- rbind(combined, Xprev)
      type_col <- c(type_col, rep("prev_inducing_now", nrow(Xprev)))
      task_col <- c(task_col, rep(t_num, nrow(Xprev)))
      spread_rows[[length(spread_rows) + 1]] <- data.frame(
        task = t_num, type = "prev_inducing_now", n = nrow(Xprev),
        mean_var = mean(diag(cov(Xprev))),
        mean_dist_to_true_center = mean(sqrt(rowSums(sweep(Xprev, 2, true_center)^2)))
      )
    }
  }
}

if (nrow(combined) < 5) stop("Not enough points collected to embed/plot")

spread_df <- do.call(rbind, spread_rows)

# ---- scale-drift baseline: mean per-dim variance of each task's NEWLY-introduced
# classes' true (never-replayed) data. The encoder keeps training across tasks, so
# raw feature-space scale can drift for every class, not just replayed ones -- this
# baseline isolates how much of class `cls`'s narrowing is IN EXCESS of that generic
# drift, vs. explained by it. ----
classes_per_task <- lapply(task_dirs_all, function(d) {
  fn <- list.files(d, pattern = "^GPparams_c[0-9]+\\.rds$")
  as.integer(sub("^GPparams_c([0-9]+)\\.rds$", "\\1", fn))
})

baseline_by_task <- setNames(rep(NA_real_, length(task_dirs_all)), as.character(task_nums_all))
prev_classes <- integer(0)
for (i in seq_along(task_dirs_all)) {
  new_classes <- setdiff(classes_per_task[[i]], prev_classes)
  if (length(new_classes) > 0) {
    tf <- paste0(task_dirs_all[i], "/train_feat.csv")
    if (file.exists(tf)) {
      tdf <- read.csv(tf)
      tdf$label <- as.numeric(as.character(tdf$label))
      per_class_var <- sapply(new_classes, function(c2) {
        rows <- as.matrix(tdf[tdf$label == c2, 1:f, drop = FALSE])
        if (nrow(rows) < 2) return(NA_real_)
        mean(diag(cov(rows)))
      })
      baseline_by_task[as.character(task_nums_all[i])] <- mean(per_class_var, na.rm = TRUE)
    }
  }
  prev_classes <- classes_per_task[[i]]
}
# forward-fill: a task that introduces no new classes inherits the last known baseline
last_val <- NA_real_
for (i in seq_along(baseline_by_task)) {
  if (is.na(baseline_by_task[i])) baseline_by_task[i] <- last_val else last_val <- baseline_by_task[i]
}

spread_df$scale_baseline <- baseline_by_task[as.character(spread_df$task)]
spread_df$normalized_var <- spread_df$mean_var / spread_df$scale_baseline

write.csv(spread_df, file = paste0(out_path, "/GP_trajectory_spread_c", cls, ".csv"), row.names = FALSE)
print(spread_df)

# ---- one shared UMAP over everything ----
umap_config <- umap.defaults
umap_config$random_state <- args$seed
umap_out <- umap(combined, config = umap_config)
proj <- umap_out$layout

all_type_levels <- c("true", "inducing", "replay", "test", "orig_now", "train_now", "prev_inducing_now")

plot_df <- data.frame(UMAP1 = proj[, 1], UMAP2 = proj[, 2],
                       type = factor(type_col, levels = all_type_levels),
                       task = task_col)

# Layers drawn on pages 1-4, bottom -> top (inducing last so it always renders on top)
main_layer_types <- c("replay", "test", "orig_now", "train_now", "inducing")
# Legend labels for pages 1-4; inducing = the Z_t saved in GPparams_c<class>.rds,
# which GP_sample.R samples next task's replay around
type_labels <- function(b) ifelse(b == "inducing", "inducing saved for next task", b)
# Pages 2/4: inducing as an opaque red cross, everything else solid circles
facet_colors <- c(replay = "steelblue", test = "darkgreen", orig_now = "purple",
                  train_now = "darkorange", inducing = "red")
facet_shapes <- c(replay = 16, test = 16, orig_now = 16, train_now = 16, inducing = 4)
facet_size   <- function(ty) if (ty == "inducing") 2.2 else 1.6
facet_stroke <- function(ty) if (ty == "inducing") 1.0 else 0.5
# orig_now / train_now are dense clouds -- draw them lighter so the layers
# beneath and the inducing points on top stay readable
dense_alpha  <- 0.3
layer_alpha  <- function(ty) if (ty %in% c("orig_now", "train_now")) dense_alpha else 0.85

pdf_path <- paste0(out_path, "/GP_visualize_trajectory_c", cls, ".pdf")
pdf(file = pdf_path, width = 9, height = 7)

# ---- Page 1: overlay -- true data (grey) + selected points colored by task, with centroid path ----
p1 <- ggplot() +
  geom_point(data = subset(plot_df, type == "true"),
             aes(x = UMAP1, y = UMAP2), color = "grey75", size = 1.0, alpha = 0.5) +
  stat_ellipse(data = subset(plot_df, type == "true"),
               aes(x = UMAP1, y = UMAP2), level = 0.95, color = "grey50",
               linewidth = 0.8, linetype = "dashed")

present_types <- intersect(main_layer_types, unique(as.character(plot_df$type)))
path_colors <- c(inducing = "firebrick", replay = "steelblue", test = "darkgreen", orig_now = "purple",
                  train_now = "darkorange", prev_inducing_now = "brown")
shape_values <- c(inducing = 21, replay = 24, test = 22, orig_now = 23,
                   train_now = 25, prev_inducing_now = 3)

for (ty in present_types) {
  sub_pts <- subset(plot_df, type == ty)
  p1 <- p1 +
    geom_point(data = sub_pts, aes(x = UMAP1, y = UMAP2, fill = task, shape = type),
               color = "black", size = 2.2, stroke = 0.3, alpha = layer_alpha(ty))
  centroids <- aggregate(cbind(UMAP1, UMAP2) ~ task, data = sub_pts, FUN = mean)
  centroids <- centroids[order(centroids$task), ]
  p1 <- p1 +
    geom_path(data = centroids, aes(x = UMAP1, y = UMAP2),
              arrow = arrow(length = unit(0.15, "inches"), type = "closed"),
              linewidth = 0.9, color = path_colors[[ty]])
}

p1 <- p1 +
  scale_fill_viridis_c(name = "task", option = "plasma") +
  scale_shape_manual(values = shape_values, labels = type_labels, name = "type") +
  labs(title = paste0("Class ", cls, ": true data (grey, task", intro_task_num,
                       ") vs. selected points across later tasks"),
       subtitle = "arrow = path of each task's centroid (red=inducing, blue=replay, orange=current training data, purple=original images re-encoded now); shared unsupervised UMAP",
       x = "UMAP1", y = "UMAP2") +
  theme_minimal()
print(p1)

# ---- Page 2: facet by task, true data as constant grey backdrop in every panel ----
facet_df <- subset(plot_df, type != "true")
if (nrow(facet_df) > 0) {
  # replicate the true/grey backdrop into every facet panel -- facet_wrap()
  # would otherwise only draw it in the introduction task's own panel, since
  # it matches each layer's data to panels by that layer's own `task` values
  true_base <- subset(plot_df, type == "true")
  facet_tasks <- sort(unique(facet_df$task))
  true_bg <- do.call(rbind, lapply(facet_tasks, function(tn) {
    d <- true_base; d$task <- tn; d
  }))
  p2 <- ggplot() +
    geom_point(data = true_bg, aes(x = UMAP1, y = UMAP2), color = "grey80", size = 0.8, alpha = 0.4)
  for (ty in intersect(main_layer_types, unique(as.character(facet_df$type)))) {
    p2 <- p2 +
      geom_point(data = subset(facet_df, type == ty), aes(x = UMAP1, y = UMAP2, color = type, shape = type),
                 size = facet_size(ty), stroke = facet_stroke(ty), alpha = layer_alpha(ty))
  }
  p2 <- p2 +
    facet_wrap(~ task, labeller = label_both) +
    scale_color_manual(values = facet_colors, labels = type_labels, name = "type") +
    scale_shape_manual(values = facet_shapes, labels = type_labels, name = "type") +
    labs(title = paste0("Class ", cls, ": selected points by task (grey = true data at introduction)"),
         x = "UMAP1", y = "UMAP2") +
    theme_minimal()
  print(p2)
}

# ---- Pages 3/4: same layers, shared LINEAR PCA instead of UMAP ----
# PCA (unlike UMAP) preserves relative distances/spread along the axes it
# keeps, so a cloud that moved looks different here from a cloud that
# shrank -- this is the view to actually judge shift vs. shrinkage from.
pca_out  <- prcomp(combined, center = TRUE, scale. = FALSE)
pca_proj <- pca_out$x[, 1:2]
pct_var  <- round(100 * (pca_out$sdev^2 / sum(pca_out$sdev^2))[1:2], 1)
print(paste0("PCA: PC1 explains ", pct_var[1], "% var, PC2 explains ", pct_var[2], "% var"))

plot_df_pca <- data.frame(PC1 = pca_proj[, 1], PC2 = pca_proj[, 2],
                           type = factor(type_col, levels = all_type_levels),
                           task = task_col)
pc1_lab <- paste0("PC1 (", pct_var[1], "% var)")
pc2_lab <- paste0("PC2 (", pct_var[2], "% var)")

# Page 3: overlay -- true data (grey) + selected points colored by task, with centroid path
p1_pca <- ggplot() +
  geom_point(data = subset(plot_df_pca, type == "true"),
             aes(x = PC1, y = PC2), color = "grey75", size = 1.0, alpha = 0.5) +
  stat_ellipse(data = subset(plot_df_pca, type == "true"),
               aes(x = PC1, y = PC2), level = 0.95, color = "grey50",
               linewidth = 0.8, linetype = "dashed")

for (ty in intersect(main_layer_types, unique(as.character(plot_df_pca$type)))) {
  sub_pts <- subset(plot_df_pca, type == ty)
  p1_pca <- p1_pca +
    geom_point(data = sub_pts, aes(x = PC1, y = PC2, fill = task, shape = type),
               color = "black", size = 2.2, stroke = 0.3, alpha = layer_alpha(ty))
  centroids <- aggregate(cbind(PC1, PC2) ~ task, data = sub_pts, FUN = mean)
  centroids <- centroids[order(centroids$task), ]
  p1_pca <- p1_pca +
    geom_path(data = centroids, aes(x = PC1, y = PC2),
              arrow = arrow(length = unit(0.15, "inches"), type = "closed"),
              linewidth = 0.9, color = path_colors[[ty]])
}

p1_pca <- p1_pca +
  scale_fill_viridis_c(name = "task", option = "plasma") +
  scale_shape_manual(values = shape_values, labels = type_labels, name = "type") +
  labs(title = paste0("Class ", cls, ": true data (grey, task", intro_task_num,
                       ") vs. selected points across later tasks (linear PCA)"),
       subtitle = "arrow = path of each task's centroid (red=inducing, blue=replay, orange=current training data, purple=original images re-encoded now); shared linear PCA -- distances are real, unlike UMAP",
       x = pc1_lab, y = pc2_lab) +
  theme_minimal()
print(p1_pca)

# Page 4: facet by task, true data as constant grey backdrop in every panel
facet_df_pca <- subset(plot_df_pca, type != "true")
if (nrow(facet_df_pca) > 0) {
  true_base_pca <- subset(plot_df_pca, type == "true")
  facet_tasks_pca <- sort(unique(facet_df_pca$task))
  true_bg_pca <- do.call(rbind, lapply(facet_tasks_pca, function(tn) {
    d <- true_base_pca; d$task <- tn; d
  }))
  p2_pca <- ggplot() +
    geom_point(data = true_bg_pca, aes(x = PC1, y = PC2), color = "grey80", size = 0.8, alpha = 0.4)
  for (ty in intersect(main_layer_types, unique(as.character(facet_df_pca$type)))) {
    p2_pca <- p2_pca +
      geom_point(data = subset(facet_df_pca, type == ty), aes(x = PC1, y = PC2, color = type, shape = type),
                 size = facet_size(ty), stroke = facet_stroke(ty), alpha = layer_alpha(ty))
  }
  p2_pca <- p2_pca +
    facet_wrap(~ task, labeller = label_both) +
    scale_color_manual(values = facet_colors, labels = type_labels, name = "type") +
    scale_shape_manual(values = facet_shapes, labels = type_labels, name = "type") +
    labs(title = paste0("Class ", cls, ": selected points by task (grey = true data at introduction, linear PCA)"),
         x = pc1_lab, y = pc2_lab) +
    theme_minimal()
  print(p2_pca)
}

# ---- Page 5/6/7: quantitative spread over tasks, in raw f-dim feature space (no UMAP/PCA distortion) ----
sel_spread <- subset(spread_df, type != "true")
if (nrow(sel_spread) > 0) {
  p3 <- ggplot(sel_spread, aes(x = task, y = mean_var, color = type)) +
    geom_hline(yintercept = spread_df$mean_var[spread_df$type == "true"],
               linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
    scale_color_manual(values = path_colors) +
    labs(title = paste0("Class ", cls, ": mean per-dimension variance over tasks (raw ", f, "-dim feature space)"),
         subtitle = "dashed line = true training data's variance at introduction task; a downward trend = narrowing coverage",
         x = "task", y = "mean variance per dimension") +
    theme_minimal()
  print(p3)

  p4 <- ggplot(sel_spread, aes(x = task, y = mean_dist_to_true_center, color = type)) +
    geom_hline(yintercept = spread_df$mean_dist_to_true_center[spread_df$type == "true"],
               linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
    scale_color_manual(values = path_colors) +
    labs(title = paste0("Class ", cls, ": mean distance to true data's centroid over tasks"),
         subtitle = "dashed line = true data's own mean distance to its centroid; a downward trend = drifting/shrinking toward a sub-region",
         x = "task", y = "mean distance to true centroid") +
    theme_minimal()
  print(p4)

  p5 <- ggplot(sel_spread, aes(x = task, y = normalized_var, color = type)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
    scale_color_manual(values = path_colors) +
    labs(title = paste0("Class ", cls, ": variance vs. that task's own encoder-scale baseline"),
         subtitle = paste0("normalized_var = mean_var / (mean_var of that task's newly-introduced classes)\n",
                            "at 1 = matches generic encoder drift; below 1 = narrowing IN EXCESS of drift"),
         x = "task", y = "mean_var / task's new-class baseline") +
    theme_minimal()
  print(p5)
}

# ---- Pages 8/9: overlap check -- current training data vs. last task's
# inducing points vs. original data, ALL re-encoded by THIS task's own
# encoder. Answers: does what GP_train.R actually trains on this task
# (train_now) still land where the GP's own support points from last task
# land once pushed through today's encoder (prev_inducing_now), and does
# either still land near the original real cluster (orig_now)? Draw order
# is bottom -> top: orig_now, then train_now (both solid, semi-transparent
# circles, so overlap between THESE TWO shows as darker regions), then
# prev_inducing_now on top as an opaque red cross, since it's a sparse set
# of specific support points we want to always be able to pick out exactly,
# not blended in. ----
overlap_types  <- c("orig_now", "train_now", "prev_inducing_now")  # draw order: bottom -> top
overlap_colors <- c(orig_now = "purple", train_now = "darkorange", prev_inducing_now = "red")
overlap_shapes <- c(orig_now = 16, train_now = 16, prev_inducing_now = 4)  # solid circle, solid circle, cross
overlap_alpha  <- c(orig_now = dense_alpha, train_now = dense_alpha, prev_inducing_now = 0.9)
overlap_size   <- c(orig_now = 1.8, train_now = 1.8, prev_inducing_now = 2.5)

overlap_present <- intersect(overlap_types, unique(as.character(plot_df$type)))
if (length(overlap_present) > 0) {
  facet_df_ov <- subset(plot_df, type %in% overlap_types)
  true_bg_ov <- do.call(rbind, lapply(sort(unique(facet_df_ov$task)), function(tn) {
    d <- subset(plot_df, type == "true"); d$task <- tn; d
  }))
  p6 <- ggplot() +
    geom_point(data = true_bg_ov, aes(x = UMAP1, y = UMAP2), color = "grey85", size = 0.6, alpha = 0.3)
  for (ty in overlap_present) {
    p6 <- p6 +
      geom_point(data = subset(facet_df_ov, type == ty),
                 aes(x = UMAP1, y = UMAP2, color = type, shape = type),
                 size = overlap_size[[ty]], stroke = 0.9, alpha = overlap_alpha[[ty]])
  }
  p6 <- p6 +
    facet_wrap(~ task, labeller = label_both) +
    scale_color_manual(values = overlap_colors, breaks = overlap_types) +
    scale_shape_manual(values = overlap_shapes, breaks = overlap_types) +
    labs(title = paste0("Class ", cls, ": current training data vs. last task's inducing points vs. original data",
                         " -- all re-encoded by THIS task's own model (UMAP)"),
         subtitle = "purple/orange solid + transparent = orig_now/train_now (overlap reads darker); red cross = prev_inducing_now (opaque, on top); grey = true data (fixed reference, faint)",
         x = "UMAP1", y = "UMAP2") +
    theme_minimal()
  print(p6)

  facet_df_ov_pca <- subset(plot_df_pca, type %in% overlap_types)
  true_bg_ov_pca <- do.call(rbind, lapply(sort(unique(facet_df_ov_pca$task)), function(tn) {
    d <- subset(plot_df_pca, type == "true"); d$task <- tn; d
  }))
  p7 <- ggplot() +
    geom_point(data = true_bg_ov_pca, aes(x = PC1, y = PC2), color = "grey85", size = 0.6, alpha = 0.3)
  for (ty in overlap_present) {
    p7 <- p7 +
      geom_point(data = subset(facet_df_ov_pca, type == ty),
                 aes(x = PC1, y = PC2, color = type, shape = type),
                 size = overlap_size[[ty]], stroke = 0.9, alpha = overlap_alpha[[ty]])
  }
  p7 <- p7 +
    facet_wrap(~ task, labeller = label_both) +
    scale_color_manual(values = overlap_colors, breaks = overlap_types) +
    scale_shape_manual(values = overlap_shapes, breaks = overlap_types) +
    labs(title = paste0("Class ", cls, ": current training data vs. last task's inducing points vs. original data",
                         " -- all re-encoded by THIS task's own model (linear PCA)"),
         subtitle = "purple/orange solid + transparent = orig_now/train_now (overlap reads darker); red cross = prev_inducing_now (opaque, on top); grey = true data (fixed reference, faint); distances are real here, unlike UMAP",
         x = pc1_lab, y = pc2_lab) +
    theme_minimal()
  print(p7)
} else {
  print("No train_now/prev_inducing_now/orig_now data collected -- skipping overlap-check pages")
}

dev.off()

print(paste0("Saved plots: ", pdf_path))
print(paste0("Saved spread summary: ", out_path, "/GP_trajectory_spread_c", cls, ".csv"))
