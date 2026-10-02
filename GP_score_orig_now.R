# GP_score_orig_now.R
#
# For ONE class, checks how that class's own GP scores the class's ORIGINAL
# real training images at every task from its introduction onward.
#
# At task t, "orig_now" = the same original real images (sampled once from
# MNIST train), encoded with task t's checkpoint (model_task<t>.pt) -- see
# reencode_true_across_tasks.py, which writes them to
# <run_root>/GP_original_reencoded_c<class>.csv. Each task's rows are scored
# with THAT task's GP for the class (task<t>/GPparams_c<class>.rds), i.e. the
# same model GP_sample.R uses to filter that task's replay candidates. If the
# replay chain has drifted away from the real class, real images score low
# and would no longer pass --score_threshold.
#
# Optionally also scores task t's own training rows for the class
# (task<t>/train_feat.csv, what the GP was actually fit on) as a reference.
#
# Outputs (in --out_path, default --run_root):
#   GP_score_orig_now_c<class>.pdf           histograms per task + summary over tasks
#   GP_score_orig_now_c<class>_summary.csv   n, mean, sd, median, quantiles, pass rate per task/type
#   GP_score_orig_now_c<class>_scores.csv    every scored point (task, type, score)
#
# Requires reencode_true_across_tasks.py to have been run for this class.
#
# Usage:
#   Rscript GP_score_orig_now.R -r runs_mnist_continual/run_20260930_225304 --class 0

library(optparse)
library(gplite)
library(laGP)
source("utils.R")   # reconstruct_laGP, ggplot2, dplyr, prepare_save_dir

option_list <- list(
  make_option(c("-r", "--run_root"),     type = "character", default = NULL,
              help = "Run directory containing task0, task1, ... subfolders"),
  make_option(c("--class"),              type = "numeric",   default = NULL,
              help = "Class label to check"),
  make_option(c("-f", "--feature_size"), type = "numeric",   default = NULL,
              help = "Feature dimension [default: f_size from config.json, else 16]"),
  make_option(c("--GP_package"),         type = "character", default = NULL,
              help = "gplite or laGP [default: GP_package from config.json, else laGP]"),
  make_option(c("--score_threshold"),    type = "numeric",   default = NULL,
              help = "Replay acceptance threshold to mark on the plots [default: GP_score_threshold from config.json, else 0.9]"),
  make_option(c("--include_train_now"),  type = "logical",   default = TRUE,
              help = "Also score task t's own train_feat.csv rows for this class as a reference [default: %default]"),
  make_option(c("--n_train_now"),        type = "numeric",   default = 500,
              help = "train_now rows to sample per task [default: %default]"),
  make_option(c("--bins"),               type = "numeric",   default = 40,
              help = "Histogram bins [default: %default]"),
  make_option(c("--out_path"),           type = "character", default = NULL,
              help = "Directory to write outputs [default: --run_root]"),
  make_option(c("--seed"),               type = "numeric",   default = 42,
              help = "Random seed [default: %default]")
)

args <- parse_args(OptionParser(option_list = option_list))
if (is.null(args$run_root)) stop("--run_root is required")
if (is.null(args$class))    stop("--class is required")
set.seed(args$seed)

cls      <- args$class
out_path <- if (is.null(args$out_path)) args$run_root else args$out_path
prepare_save_dir(out_path)

# ---- defaults from config.json (written by CL_Driver.py) ----
cfg <- list()
config_json_file <- file.path(args$run_root, "config.json")
if (file.exists(config_json_file)) cfg <- jsonlite::fromJSON(config_json_file)
pick <- function(arg, cfg_val, fallback) {
  if (!is.null(arg)) arg else if (!is.null(cfg_val)) cfg_val else fallback
}
f               <- as.integer(pick(args$feature_size, cfg$f_size, 16))
GP_package      <- pick(args$GP_package, cfg$GP_package, "laGP")
score_threshold <- as.numeric(pick(args$score_threshold, cfg$GP_score_threshold, 0.9))
print(paste0("GP_score_orig_now: class=", cls, ", f=", f, ", GP_package=", GP_package,
             ", score_threshold=", score_threshold))

# ---- orig_now features ----
orig_file <- file.path(args$run_root, paste0("GP_original_reencoded_c", cls, ".csv"))
if (!file.exists(orig_file)) {
  stop(paste0(orig_file, " not found. Generate it with: python reencode_true_across_tasks.py --run_root ",
              args$run_root, " --class ", cls, " --f_size ", f))
}
orig_df <- read.csv(orig_file)

# ---- task dirs from the class's introduction onward ----
task_dirs <- list.dirs(args$run_root, recursive = FALSE)
task_dirs <- task_dirs[grepl("/task[0-9]+$", task_dirs)]
task_nums <- as.integer(sub(".*/task([0-9]+)$", "\\1", task_dirs))
ord       <- order(task_nums)
task_dirs <- task_dirs[ord]
task_nums <- task_nums[ord]
has_gp    <- file.exists(file.path(task_dirs, paste0("GPparams_c", cls, ".rds")))
if (!any(has_gp)) stop(paste0("Class ", cls, " never appears (no GPparams_c", cls, ".rds)"))
intro_task <- task_nums[min(which(has_gp))]
print(paste0("Class ", cls, " introduced at task", intro_task))

load_gp <- function(t_dir) {
  key <- paste0("c", cls)
  if (GP_package == "gplite") {
    gp_load(file.path(t_dir, paste0("GPmodel_", key, ".rda")))
  } else if (GP_package == "laGP") {
    params <- readRDS(file.path(t_dir, paste0("GPparams_", key, ".rds")))
    reconstruct_laGP(params$Z_t, params$Y_Z_t, label = cls)
  } else {
    stop(paste("Unknown GP_package:", GP_package))
  }
}
gp_score <- function(gp_model, X) {
  if (GP_package == "gplite") {
    as.numeric(gp_pred(gp_model, X, jitter = 1e-4)$mean)
  } else {
    as.numeric(predGPsep(gp_model, X)$mean)
  }
}

# ---- score every task ----
score_rows <- list()
for (i in which(has_gp)) {
  t_dir <- task_dirs[i]
  t_num <- task_nums[i]

  X_orig <- as.matrix(orig_df[orig_df$task == t_num, 1:f, drop = FALSE])
  if (nrow(X_orig) == 0) {
    print(paste0("  task", t_num, ": no orig_now rows (no checkpoint for this task?) -- skipping"))
    next
  }

  gp_model <- load_gp(t_dir)
  score_rows[[length(score_rows) + 1]] <- data.frame(
    task = t_num, type = "orig_now", score = gp_score(gp_model, X_orig))

  if (isTRUE(args$include_train_now)) {
    train_file <- file.path(t_dir, "train_feat.csv")
    if (file.exists(train_file)) {
      tr <- read.csv(train_file)
      tr <- tr[as.numeric(as.character(tr$label)) == cls, 1:f, drop = FALSE]
      if (nrow(tr) > 0) {
        tr <- tr[sample(nrow(tr), min(as.integer(args$n_train_now), nrow(tr))), , drop = FALSE]
        score_rows[[length(score_rows) + 1]] <- data.frame(
          task = t_num, type = "train_now", score = gp_score(gp_model, as.matrix(tr)))
      }
    }
  }

  if (GP_package == "laGP") deleteGPsep(gp_model)
  print(paste0("  task", t_num, ": scored ", nrow(X_orig), " orig_now points"))
}
if (length(score_rows) == 0) stop("Nothing scored -- check that the orig_now CSV covers these tasks")

scores_df <- do.call(rbind, score_rows)
scores_df$type <- factor(scores_df$type, levels = c("orig_now", "train_now"))

# ---- summary stats per task/type ----
summary_df <- do.call(rbind, lapply(split(scores_df, list(scores_df$task, scores_df$type), drop = TRUE), function(d) {
  s <- d$score
  data.frame(task = d$task[1], type = d$type[1], n = length(s),
             mean = mean(s), sd = sd(s), median = median(s),
             q05 = unname(quantile(s, 0.05)), q95 = unname(quantile(s, 0.95)),
             min = min(s), max = max(s),
             pass_rate = mean(s >= score_threshold))
}))
summary_df <- summary_df[order(summary_df$type, summary_df$task), ]
rownames(summary_df) <- NULL
print(summary_df)

write.csv(scores_df,  file.path(out_path, paste0("GP_score_orig_now_c", cls, "_scores.csv")),  row.names = FALSE)
write.csv(summary_df, file.path(out_path, paste0("GP_score_orig_now_c", cls, "_summary.csv")), row.names = FALSE)

# ---- plots ----
type_colors <- c(orig_now = "purple", train_now = "darkorange")
orig_sum    <- subset(summary_df, type == "orig_now")
orig_sum$label <- sprintf("n = %d\nmean = %.3f\nsd = %.3f\nmedian = %.3f\n>= %.2f: %.1f%%",
                          orig_sum$n, orig_sum$mean, orig_sum$sd, orig_sum$median,
                          score_threshold, 100 * orig_sum$pass_rate)

pdf_path <- file.path(out_path, paste0("GP_score_orig_now_c", cls, ".pdf"))
pdf(file = pdf_path, width = 10, height = 7)

# Page 1: orig_now score histogram per task, with stats
p1 <- ggplot(subset(scores_df, type == "orig_now"), aes(x = score)) +
  geom_histogram(bins = args$bins, fill = "purple", alpha = 0.6, color = "white") +
  geom_vline(data = orig_sum, aes(xintercept = mean), color = "black", linewidth = 0.8) +
  geom_vline(data = orig_sum, aes(xintercept = mean - sd), color = "black", linetype = "dotted") +
  geom_vline(data = orig_sum, aes(xintercept = mean + sd), color = "black", linetype = "dotted") +
  geom_vline(xintercept = score_threshold, color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_label(data = orig_sum, aes(x = -Inf, y = Inf, label = label),
             hjust = -0.05, vjust = 1.1, size = 3, label.size = 0, alpha = 0.8) +
  facet_wrap(~ task, labeller = label_both) +
  labs(title = paste0("Class ", cls, ": GP score of original real images (orig_now), per task"),
       subtitle = paste0("Task t: original class-", cls, " images encoded with task t's encoder, scored by task t's class-",
                         cls, " GP\nsolid = mean, dotted = mean +/- sd, red dashed = replay threshold (",
                         score_threshold, ")"),
       x = "GP score", y = "count") +
  theme_minimal()
print(p1)

# Page 2: orig_now vs. train_now (what the GP was fit on), per task
if (any(scores_df$type == "train_now")) {
  p2 <- ggplot(scores_df, aes(x = score, fill = type)) +
    geom_histogram(bins = args$bins, alpha = 0.5, position = "identity", color = NA) +
    geom_vline(xintercept = score_threshold, color = "red", linetype = "dashed", linewidth = 0.8) +
    facet_wrap(~ task, labeller = label_both, scales = "free_y") +
    scale_fill_manual(values = type_colors) +
    labs(title = paste0("Class ", cls, ": GP score of original images (orig_now) vs. task's own training data (train_now)"),
         subtitle = paste0("train_now = task t's train_feat.csv rows for class ", cls,
                           " (real at introduction, decoded replay afterward); red dashed = replay threshold"),
         x = "GP score", y = "count") +
    theme_minimal()
  print(p2)
}

# Page 3: mean +/- sd and pass rate over tasks
p3 <- ggplot(summary_df, aes(x = task, y = mean, color = type)) +
  geom_hline(yintercept = score_threshold, color = "red", linetype = "dashed") +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.15) +
  geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
  scale_color_manual(values = type_colors) +
  labs(title = paste0("Class ", cls, ": mean GP score over tasks (error bar = +/- sd)"),
       subtitle = "red dashed = replay threshold", x = "task", y = "GP score") +
  theme_minimal()
print(p3)

p4 <- ggplot(summary_df, aes(x = task, y = 100 * pass_rate, color = type)) +
  geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
  scale_color_manual(values = type_colors) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = paste0("Class ", cls, ": % of points scoring >= replay threshold (", score_threshold, ")"),
       subtitle = "i.e. how many would survive GP_sample.R's filter at that task",
       x = "task", y = "% >= threshold") +
  theme_minimal()
print(p4)

dev.off()
print(paste0("Saved plots: ", pdf_path))
print(paste0("Saved summary: ", file.path(out_path, paste0("GP_score_orig_now_c", cls, "_summary.csv"))))
