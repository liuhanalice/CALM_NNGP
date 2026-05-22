# check_gp_on_replay.R
#
# Loads all GP models for a task directory, runs argmax prediction on
# replay_points.csv, writes predictions to replay_gp_preds.csv, and
# prints per-class confusion stats to stdout.
#
# Usage (called by check_replay_labels.py):
#   Rscript check_gp_on_replay.R \
#     --save_path runs/run_xxx/task1 \
#     --existing_classes 0,1,2,3,4,6 \
#     --feature_size 16 \
#     --GP_package laGP

library(optparse)
library(gplite)
library(laGP)
source("utils.R")

option_list <- list(
  make_option(c("-p", "--save_path"),      type = "character", default = NULL,
              help = "Task directory (contains GPparams_*.rds and replay_points.csv)"),
  make_option(c("--existing_classes"),     type = "character", default = "0,1,2,3,4",
              help = "Comma-separated class labels seen at this task"),
  make_option(c("-f", "--feature_size"),   type = "numeric",   default = 16,
              help = "Feature dimension [default: %default]"),
  make_option(c("--GP_package"),           type = "character", default = "laGP",
              help = "GP package: gplite or laGP [default: %default]"),
  make_option(c("--out_csv"),              type = "character", default = NULL,
              help = "Output predictions CSV (default: <save_path>/replay_gp_preds.csv)")
)

parser <- OptionParser(option_list = option_list)
args   <- parse_args(parser)

if (is.null(args$save_path)) stop("--save_path is required")
f               <- as.integer(args$feature_size)
GP_package      <- args$GP_package
existing_classes <- as.list(as.numeric(strsplit(args$existing_classes, ",")[[1]]))
out_csv <- if (!is.null(args$out_csv)) args$out_csv else
           file.path(args$save_path, "replay_gp_preds.csv")

# ---------- load GP models ----------
GPmodels <- list()
for (label in unlist(existing_classes)) {
  key        <- paste0("c", label)
  params_file <- file.path(args$save_path, paste0("GPparams_", key, ".rds"))
  if (!file.exists(params_file)) {
    stop(paste0("GPparams not found: ", params_file))
  }
  params <- readRDS(params_file)

  if (GP_package == "gplite") {
    rda_file <- file.path(args$save_path, paste0("GPmodel_", key, ".rda"))
    if (!file.exists(rda_file)) stop(paste0("GPmodel rda not found: ", rda_file))
    GPmodels[[key]] <- gp_load(rda_file)
  } else if (GP_package == "laGP") {
    da <- darg(list(mle = TRUE), params$Z_t)
    ga <- tryCatch(
      garg(list(mle = TRUE), matrix(params$Y_Z_t)),
      error = function(e) list(start = 1e-3, min = sqrt(.Machine$double.eps), max = 1.0)
    )
    gp_model <- newGPsep(X = params$Z_t, Z = params$Y_Z_t,
                         d = rep(da$start, ncol(params$Z_t)),
                         g = ga$start, dK = TRUE)
    mleGPsep(gp_model, param = "both",
             tmin = c(da$min, ga$min),
             tmax = c(da$max, ga$max))
    GPmodels[[key]] <- gp_model
  } else {
    stop(paste0("Unknown GP_package: ", GP_package))
  }
  cat(sprintf("  Loaded GP for class %d\n", label))
}

# ---------- load replay points ----------
replay_csv <- file.path(args$save_path, "replay_points.csv")
if (!file.exists(replay_csv)) stop(paste0("replay_points.csv not found: ", replay_csv))
replay_df  <- read.csv(replay_csv)
feat_cols  <- paste0("f", 0:(f - 1))
X_replay   <- as.matrix(replay_df[, feat_cols])
true_labels <- as.integer(replay_df$label)

# ---------- GP argmax prediction ----------
score_mat <- matrix(NA_real_, nrow = nrow(X_replay), ncol = length(existing_classes))
colnames(score_mat) <- paste0("c", unlist(existing_classes))

for (j in seq_along(existing_classes)) {
  label <- existing_classes[[j]]
  key   <- paste0("c", label)
  if (GP_package == "gplite") {
    m <- gp_pred(GPmodels[[key]], X_replay, jitter = 1e-4)$mean
  } else {
    m <- predGPsep(GPmodels[[key]], X_replay)$mean
  }
  score_mat[, j] <- as.numeric(m)
}

pred_labels <- sapply(1:nrow(score_mat), function(i)
  unlist(existing_classes)[which.max(score_mat[i, ])]
)
pred_labels <- as.integer(pred_labels)

# ---------- save predictions ----------
out_df <- data.frame(true_label = true_labels, pred_label = pred_labels)
write.csv(out_df, out_csv, row.names = FALSE)
cat(sprintf("GP predictions saved -> %s\n", out_csv))

# ---------- print per-class confusion stats ----------
total      <- length(true_labels)
n_correct  <- sum(pred_labels == true_labels)
cat(sprintf("GP overall accuracy: %d/%d = %.2f%%\n", n_correct, total, 100 * n_correct / total))

classes <- sort(unique(true_labels))
cat(sprintf("  %6s  %5s  %8s  %7s\n", "Class", "N", "Correct", "Acc"))
for (c in classes) {
  mask    <- true_labels == c
  n       <- sum(mask)
  correct <- sum(pred_labels[mask] == c)
  cat(sprintf("  %6d  %5d  %8d  %6.1f%%\n", c, n, correct, 100 * correct / n))
}

# misclassifications
wrong_mask <- pred_labels != true_labels
if (any(wrong_mask)) {
  cat(sprintf("\nGP Misclassifications (%d):\n", sum(wrong_mask)))
  for (c in classes) {
    mask_c <- true_labels == c & wrong_mask
    if (!any(mask_c)) next
    wrong_preds <- pred_labels[mask_c]
    tbl <- table(wrong_preds)
    pred_str <- paste(sprintf("%s(%d)", names(tbl), as.integer(tbl)), collapse = ", ")
    cat(sprintf("  True=%d -> predicted as: %s\n", c, pred_str))
  }
}
