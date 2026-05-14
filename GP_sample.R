# GP_sample.R
# Loads trained GP models saved by GP_train.R and generates replay-buffer points
# by sampling from the class distribution (center + covariance) and keeping only
# candidates whose GP score is >= score_threshold (resampling until enough are found).
#
# Supports --GP_package gplite | laGP  (same flag as GP_train.R)
#
# gplite: loads GPmodel_{key}.rda via gp_load
# laGP:   cannot persist C pointers; reconstructs GP from GPparams_{key}.rds
#         (Z_t, Y_Z_t saved by GP_train.R)

library(optparse)
library(gplite)
library(laGP)
library(MASS)
source("utils.R")

option_list <- list(
  make_option(c("-f", "--feature_size"), type = "numeric", default = 16,
              help = "Feature dimension (must match GP_train.R) [default: %default]"),
  make_option(c("-p", "--save_path"),    type = "character", default = NULL,
              help = "Directory written by GP_train.R (contains GPparams_*.rds)"),
  make_option(c("--existing_classes"),   type = "character", default = "0,1",
              help = "Comma-separated class labels, same as GP_train.R [default: %default]"),
  make_option(c("--n_indcpts"),          type = "numeric",   default = 1000,
              help = "GP-sampled points to keep per class [default: %default]"),
  make_option(c("--GP_package"),         type = "character", default = "gplite",
              help = "GP package used in GP_train.R: gplite or laGP [default: %default]"),
  make_option(c("--n_cand_mult"),        type = "numeric",   default = 10,
              help = "Candidates per resample iteration = n_indcpts * n_cand_mult [default: %default]"),
  make_option(c("--score_threshold"),    type = "numeric",   default = 0.9,
              help = "Minimum GP score for a sample to be kept [default: %default]"),
  make_option(c("--max_resample_iter"),  type = "numeric",   default = 50,
              help = "Maximum resample iterations per class [default: %default]"),
  make_option(c("--seed"),               type = "numeric",   default = 42,
              help = "Random seed [default: %default]")
)

parser <- OptionParser(option_list = option_list)
args   <- parse_args(parser)

set.seed(args$seed)

if (is.null(args$save_path)) stop("--save_path is required")

f                <- as.integer(args$feature_size)
num_indcpts      <- as.integer(args$n_indcpts)
n_cand_per_iter  <- as.integer(num_indcpts * args$n_cand_mult)
score_threshold  <- args$score_threshold
max_resample_iter <- as.integer(args$max_resample_iter)
GP_package       <- args$GP_package
existing_classes <- as.list(as.numeric(strsplit(args$existing_classes, ",")[[1]]))

print(paste0("GP_sample: package=", GP_package,
             ", classes=", args$existing_classes,
             ", n_indcpts=", num_indcpts,
             ", n_cand_per_iter=", n_cand_per_iter,
             ", score_threshold=", score_threshold,
             ", max_resample_iter=", max_resample_iter))

replay_all <- matrix(NA_real_, nrow = 0, ncol = f)
labels_all   <- integer(0)

for (j in seq_along(existing_classes)) {
  label <- existing_classes[[j]]
  key   <- paste0("c", label)

  params_file <- paste0(args$save_path, "/GPparams_", key, ".rds")
  if (!file.exists(params_file)) stop(paste("Missing params file:", params_file))
  params <- readRDS(params_file)

  if (is.null(params$center) || is.null(params$covariance)) {
    stop(paste0("GPparams_", key, ".rds is missing center/covariance — re-run GP_train.R"))
  }

  # ---- Load / reconstruct GP model ----
  if (GP_package == "gplite") {
    model_file <- paste0(args$save_path, "/GPmodel_", key, ".rda")
    if (!file.exists(model_file)) stop(paste("Missing model file:", model_file))
    gp_model <- gp_load(model_file)

  } else if (GP_package == "laGP") {
    da <- darg(list(mle = TRUE), params$Z_t)
    ga <- tryCatch(
      garg(list(mle = TRUE), matrix(params$Y_Z_t)),
      error = function(e) {
        print(paste0("  garg failed (near-constant Y_Z_t, var=",
                     round(var(as.numeric(params$Y_Z_t)), 8),
                     ") - using default g bounds"))
        list(start = 1e-3, min = sqrt(.Machine$double.eps), max = 1.0)
      }
    )
    gp_model <- newGPsep(X = params$Z_t, Z = params$Y_Z_t,
                         d = rep(da$start, ncol(params$Z_t)),
                         g = ga$start, dK = TRUE)
    mleGPsep(gp_model, param = "both",
             tmin = c(da$min, ga$min),
             tmax = c(da$max, ga$max))

  } else {
    stop(paste("Unknown GP_package:", GP_package))
  }

  # ---- Sample from class distribution and filter by GP score ----
  center     <- params$center
  # Small diagonal jitter for numerical stability of mvrnorm
  covariance <- params$covariance + diag(1e-8, nrow(params$covariance))

  print(paste0("Class ", label, ": sampling from class distribution (center + covariance)"))

  collected      <- matrix(NA_real_, nrow = 0, ncol = f)
  kept_scores    <- numeric(0)
  iter           <- 0

  while (nrow(collected) < num_indcpts && iter < max_resample_iter) {
    iter   <- iter + 1
    X_cand <- mvrnorm(n_cand_per_iter, mu = center, Sigma = covariance)

    if (GP_package == "gplite") {
      scores <- as.numeric(gp_pred(gp_model, X_cand, jitter = 1e-4)$mean)
    } else {
      scores <- as.numeric(predGPsep(gp_model, X_cand)$mean)
    }

    keep <- which(scores >= score_threshold)
    if (length(keep) > 0) {
      collected   <- rbind(collected, X_cand[keep, , drop = FALSE])
      kept_scores <- c(kept_scores, scores[keep])
    }
    print(paste0("  iter ", iter, ": ", length(keep), "/", n_cand_per_iter,
                 " passed score >= ", score_threshold,
                 " (collected ", nrow(collected), "/", num_indcpts, ")"))
  }

  if (GP_package == "laGP") deleteGPsep(gp_model)

  if (nrow(collected) == 0) {
    warning(paste0("Class ", label, ": no samples passed threshold after ",
                   max_resample_iter, " iterations — skipping"))
    next
  }

  # Cap to num_indcpts if we collected more
  if (nrow(collected) > num_indcpts) {
    sel        <- sample(nrow(collected), num_indcpts)
    collected  <- collected[sel, , drop = FALSE]
    kept_scores <- kept_scores[sel]
  }

  print(paste0("  Class ", label, ": kept ", nrow(collected), " points",
               " | score range [", round(min(kept_scores), 4),
               ", ", round(max(kept_scores), 4), "]"))

  replay_all <- rbind(replay_all, collected)
  labels_all   <- c(labels_all, rep(as.integer(label), nrow(collected)))
}

# ---- Write replay_points.csv ----
out_df <- as.data.frame(replay_all)
colnames(out_df) <- paste0("f", seq_len(f) - 1)
out_df$label <- labels_all

out_path <- paste0(args$save_path, "/replay_points.csv")
write.csv(out_df, file = out_path, row.names = FALSE)
print(paste0("GP-sampled replay_points.csv written: ",
             nrow(out_df), " total rows -> ", out_path))
