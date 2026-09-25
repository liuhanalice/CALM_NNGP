# GP_sample.R
# Loads trained GP models saved by GP_train.R and generates replay-buffer points
# by drawing candidates and keeping only those whose GP score is >= score_threshold
# (resampling until enough are found).
#
# --sample_mode class_dist | inducing
#   class_dist: candidates ~ N(center, covariance) of the class's real features
#   inducing:   candidates ~ N(z_i, inducing_noise_scale^2 * covariance) around each
#               true-class inducing point z_i, i.e. those whose GP score is
#               >= inducing_score_min (Z_t also stores other-class inducing points
#               with score ~0, which are excluded). Balanced: each of the m anchors
#               gets a quota of n_replay %/% m kept points (the n_replay %% m
#               remainder goes to randomly chosen anchors), and the quota is
#               filled per anchor after the GP-score filter.
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
  make_option(c("--n_replay"),           type = "numeric",   default = 1000,
              help = "Replay points to keep per class [default: %default]"),
  make_option(c("--GP_package"),         type = "character", default = "gplite",
              help = "GP package used in GP_train.R: gplite or laGP [default: %default]"),
  make_option(c("--score_threshold"),    type = "numeric",   default = 0.9,
              help = "Minimum GP score for a sample to be kept [default: %default]"),
  make_option(c("--sample_mode"),        type = "character", default = "class_dist",
              help = "Candidate source: class_dist (N(center, cov)) or inducing (Gaussian around true-class inducing points) [default: %default]"),
  make_option(c("--inducing_score_min"), type = "numeric",   default = 0.5,
              help = "inducing mode: min GP score for an inducing point to count as true-class [default: %default]"),
  make_option(c("--inducing_noise_scale"), type = "numeric", default = 0.1,
              help = "inducing mode: perturbation covariance = scale^2 * class covariance [default: %default]"),
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
num_replay      <- as.integer(args$n_replay)

score_threshold  <- args$score_threshold
max_resample_iter <- as.integer(args$max_resample_iter)
GP_package       <- args$GP_package
sample_mode      <- args$sample_mode
if (!sample_mode %in% c("class_dist", "inducing")) stop(paste("Unknown sample_mode:", sample_mode))
inducing_score_min   <- if (is.null(args$inducing_score_min)) score_threshold else args$inducing_score_min
inducing_noise_scale <- args$inducing_noise_scale
existing_classes <- as.list(as.numeric(strsplit(args$existing_classes, ",")[[1]]))

print(paste0("GP_sample: package=", GP_package,
             ", classes=", args$existing_classes,
             ", n_replay=", num_replay,
             ", score_threshold=", score_threshold,
             ", max_resample_iter=", max_resample_iter,
             ", sample_mode=", sample_mode,
             if (sample_mode == "inducing")
               paste0(", inducing_score_min=", inducing_score_min,
                      ", inducing_noise_scale=", inducing_noise_scale) else ""))

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
    gp_model <- reconstruct_laGP(params$Z_t, params$Y_Z_t, label = label)

  } else {
    stop(paste("Unknown GP_package:", GP_package))
  }

  gp_score <- function(X) {
    if (GP_package == "gplite") {
      as.numeric(gp_pred(gp_model, X, jitter = 1e-4)$mean)
    } else {
      as.numeric(predGPsep(gp_model, X)$mean)
    }
  }

  # ---- Draw candidates around anchors and filter by GP score ----
  # Each anchor a_i has a quota q_i of kept points; candidates are
  # a_i + N(0, noise_cov). class_dist is the single-anchor case (center, covariance).
  center     <- params$center
  # Small diagonal jitter for numerical stability of mvrnorm
  covariance <- params$covariance + diag(1e-8, nrow(params$covariance))

  if (sample_mode == "class_dist") {
    print(paste0("Class ", label, ": sampling from class distribution (center + covariance)"))
    anchors   <- matrix(center, nrow = 1)
    noise_cov <- covariance

  } else {
    # Keep only true-class inducing points (Z_t also holds other-class points, score ~0)
    Z          <- as.matrix(params$Z_t)
    Z_scores   <- gp_score(Z)
    anchors    <- Z[Z_scores >= inducing_score_min, , drop = FALSE]
    print(paste0("Class ", label, ": sampling around ", nrow(anchors), "/", nrow(Z),
                 " inducing points with score >= ", inducing_score_min))
    if (nrow(anchors) == 0) {
      if (GP_package == "laGP") deleteGPsep(gp_model)
      warning(paste0("Class ", label, ": no inducing points with score >= ",
                     inducing_score_min, " — skipping"))
      next
    }
    noise_cov <- inducing_noise_scale^2 * covariance
  }

  # Balanced quotas: n %/% m each, remainder to randomly chosen anchors
  n_anchor <- nrow(anchors)
  quota    <- rep(num_replay %/% n_anchor, n_anchor)
  n_extra  <- num_replay %% n_anchor
  if (n_extra > 0) {
    extra        <- sample(n_anchor, n_extra)
    quota[extra] <- quota[extra] + 1L
  }
  if (sample_mode == "inducing") {
    print(paste0("  quota per inducing point: ", min(quota),
                 if (max(quota) > min(quota)) paste0("-", max(quota)) else ""))
  }

  count       <- integer(n_anchor)
  collected   <- matrix(NA_real_, nrow = 0, ncol = f)
  kept_scores <- numeric(0)
  iter        <- 0

  while (any(count < quota) && iter < max_resample_iter) {
    iter   <- iter + 1
    # Every unfilled anchor draws its full quota of candidates this round
    active <- which(count < quota)
    idx    <- rep(active, times = quota[active])
    X_cand <- anchors[idx, , drop = FALSE] +
      matrix(mvrnorm(length(idx), mu = rep(0, f), Sigma = noise_cov), ncol = f)
    scores <- gp_score(X_cand)

    # Keep passing candidates, at most the remaining quota per anchor
    passed <- which(scores >= score_threshold)
    rank   <- ave(passed, idx[passed], FUN = seq_along)
    keep   <- passed[rank <= (quota - count)[idx[passed]]]
    if (length(keep) > 0) {
      collected   <- rbind(collected, X_cand[keep, , drop = FALSE])
      kept_scores <- c(kept_scores, scores[keep])
      count       <- count + tabulate(idx[keep], nbins = n_anchor)
    }
    print(paste0("  iter ", iter, ": ", length(passed), "/", length(idx),
                 " passed score >= ", score_threshold,
                 " (collected ", sum(count), "/", num_replay,
                 if (sample_mode == "inducing")
                   paste0(", ", sum(count >= quota), "/", n_anchor, " inducing points full")
                 else "", ")"))
  }

  if (GP_package == "laGP") deleteGPsep(gp_model)

  if (nrow(collected) == 0) {
    warning(paste0("Class ", label, ": no samples passed threshold after ",
                   max_resample_iter, " iterations — skipping"))
    next
  }

  if (sample_mode == "inducing" && any(count < quota)) {
    warning(paste0("Class ", label, ": ", sum(count < quota), " inducing point(s) did not fill ",
                   "their quota after ", max_resample_iter, " iterations (short by ",
                   sum(quota - count), " points)"))
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
