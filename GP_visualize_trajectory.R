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
# Two views:
#   1. VISUAL: one shared, UNSUPERVISED UMAP embedding (fit jointly over the
#      true introduction-task data + every later task's inducing points
#      and/or replay buffer for this class) -- an overlay with a centroid
#      path per task, and a small-multiples facet by task.
#   2. QUANTITATIVE: mean per-dimension variance and mean distance to the
#      true data's centroid, computed directly in raw feature space (not
#      UMAP-distorted), plotted over task index against the true data's own
#      values as a dashed baseline.
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

# ---- collect inducing / replay points from every task from introduction onward ----
combined <- true_X
type_col <- rep("true", nrow(true_X))
task_col <- rep(intro_task_num, nrow(true_X))

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

plot_df <- data.frame(UMAP1 = proj[, 1], UMAP2 = proj[, 2],
                       type = factor(type_col, levels = c("true", "inducing", "replay")),
                       task = task_col)

pdf_path <- paste0(out_path, "/GP_visualize_trajectory_c", cls, ".pdf")
pdf(file = pdf_path, width = 9, height = 7)

# ---- Page 1: overlay -- true data (grey) + selected points colored by task, with centroid path ----
p1 <- ggplot() +
  geom_point(data = subset(plot_df, type == "true"),
             aes(x = UMAP1, y = UMAP2), color = "grey75", size = 1.0, alpha = 0.5) +
  stat_ellipse(data = subset(plot_df, type == "true"),
               aes(x = UMAP1, y = UMAP2), level = 0.95, color = "grey50",
               linewidth = 0.8, linetype = "dashed")

# draw replay first, inducing last so inducing points always render on top
present_types <- intersect(c("replay", "inducing"), unique(as.character(plot_df$type)))
path_colors <- c(inducing = "firebrick", replay = "steelblue")

for (ty in present_types) {
  sub_pts <- subset(plot_df, type == ty)
  p1 <- p1 +
    geom_point(data = sub_pts, aes(x = UMAP1, y = UMAP2, fill = task, shape = type),
               color = "black", size = 2.2, stroke = 0.3, alpha = 0.85)
  centroids <- aggregate(cbind(UMAP1, UMAP2) ~ task, data = sub_pts, FUN = mean)
  centroids <- centroids[order(centroids$task), ]
  p1 <- p1 +
    geom_path(data = centroids, aes(x = UMAP1, y = UMAP2),
              arrow = arrow(length = unit(0.15, "inches"), type = "closed"),
              linewidth = 0.9, color = path_colors[[ty]])
}

p1 <- p1 +
  scale_fill_viridis_c(name = "task", option = "plasma") +
  scale_shape_manual(values = c(inducing = 21, replay = 24), name = "type") +
  labs(title = paste0("Class ", cls, ": true data (grey, task", intro_task_num,
                       ") vs. selected points across later tasks"),
       subtitle = "arrow = path of each task's centroid (red=inducing, blue=replay); shared unsupervised UMAP",
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
  # draw replay first, inducing last so inducing points always render on top
  for (ty in intersect(c("replay", "inducing"), unique(as.character(facet_df$type)))) {
    p2 <- p2 +
      geom_point(data = subset(facet_df, type == ty), aes(x = UMAP1, y = UMAP2, color = type),
                 size = 1.6, alpha = 0.85)
  }
  p2 <- p2 +
    facet_wrap(~ task, labeller = label_both) +
    scale_color_manual(values = path_colors) +
    labs(title = paste0("Class ", cls, ": selected points by task (grey = true data at introduction)"),
         x = "UMAP1", y = "UMAP2") +
    theme_minimal()
  print(p2)
}

# ---- Page 3/4: quantitative spread over tasks, in raw f-dim feature space (no UMAP distortion) ----
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

dev.off()

print(paste0("Saved plots: ", pdf_path))
print(paste0("Saved spread summary: ", out_path, "/GP_trajectory_spread_c", cls, ".csv"))
