# GP_sample.R
# Loads trained GP models saved by GP_train.R and generates replay-buffer points
# using Strategy 3 (Perturb + GP-Filter):
#   1. Perturb seed inducing points with Gaussian noise -> candidate X*
#   2. Score each candidate with the class GP (predicts softmax score)
#   3. Keep top n_indcpts per class
# Overwrites inducing_points.csv with GP-sampled points.
#
# Supports --GP_package gplite | laGP  (same flag as GP_train.R)
#
# gplite: loads GPmodel_{key}.rda via gp_load; seeds from model$method$inducing
# laGP:   cannot persist C pointers; reconstructs GP from GPparams_{key}.rds
#         (d, g, X_train, Y_train saved by GP_train.R)

library(optparse)
library(gplite)
library(laGP)
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
  make_option(c("--sigma_perturb"),      type = "numeric",   default = 0.05,
              help = "Std of Gaussian perturbation noise added to seed points [default: %default]"),
  make_option(c("--n_cand_mult"),        type = "numeric",   default = 10,
              help = "n_candidates = n_indcpts * n_cand_mult [default: %default]"),
  make_option(c("--seed"),               type = "numeric",   default = 42,
              help = "Random seed [default: %default]")
)

parser <- OptionParser(option_list = option_list)
args   <- parse_args(parser)

set.seed(args$seed)

if (is.null(args$save_path)) stop("--save_path is required")

f                <- as.integer(args$feature_size)
num_indcpts      <- as.integer(args$n_indcpts)
n_candidates     <- as.integer(num_indcpts * args$n_cand_mult)
sigma_perturb    <- args$sigma_perturb
GP_package       <- args$GP_package
existing_classes <- as.list(as.numeric(strsplit(args$existing_classes, ",")[[1]]))

print(paste0("GP_sample: package=", GP_package,
             ", classes=", args$existing_classes,
             ", n_indcpts=", num_indcpts,
             ", n_candidates=", n_candidates,
             ", sigma_perturb=", sigma_perturb))

replay_all <- matrix(NA_real_, nrow = 0, ncol = f)
labels_all   <- integer(0)

for (j in seq_along(existing_classes)) {
  label <- existing_classes[[j]]
  key   <- paste0("c", label)

  params_file <- paste0(args$save_path, "/GPparams_", key, ".rds")
  if (!file.exists(params_file)) stop(paste("Missing params file:", params_file))
  params <- readRDS(params_file)

  # ---- Load / reconstruct GP model ----
  if (GP_package == "gplite") {
    model_file <- paste0(args$save_path, "/GPmodel_", key, ".rda")
    if (!file.exists(model_file)) stop(paste("Missing model file:", model_file))
    gp_model <- gp_load(model_file)
    # Use the (potentially optimized) inducing locations stored inside the model
    Z_seed <- gp_model$method$inducing

  } else if (GP_package == "laGP") {
    # Reconstruct a small GP on (Z_t, Y_Z_t): inducing locations as inputs,
    # full-GP posterior means at those locations as pseudo-targets.
    # Same fitted hyperparameters (d, g) -> same kernel shape, no re-fitting needed.
    gp_model <- newGPsep(X = params$Z_t, Z = params$Y_Z_t,
                         d = as.numeric(params$d), g = as.numeric(params$g), dK = FALSE)
    Z_seed <- params$Z_t

  } else {
    stop(paste("Unknown GP_package:", GP_package))
  }

  print(paste0("Class ", label, ": seeding from ", nrow(Z_seed), " inducing points"))

  # ---- Perturb seed points to generate candidates ----
  base_idx <- sample(nrow(Z_seed), n_candidates, replace = TRUE)
  noise    <- matrix(rnorm(n_candidates * f, sd = sigma_perturb),
                     nrow = n_candidates, ncol = f)
  X_cand   <- Z_seed[base_idx, , drop = FALSE] + noise

  # ---- Score candidates with the class GP ----
  if (GP_package == "gplite") {
    pred   <- gp_pred(gp_model, X_cand, jitter = 1e-4)
    scores <- as.numeric(pred$mean)
  } else {
    pred   <- predGPsep(gp_model, X_cand)
    scores <- as.numeric(pred$mean)
    deleteGPsep(gp_model)   # free C memory immediately after use
  }

  # ---- Keep top n_indcpts by GP confidence score ----
  k    <- min(num_indcpts, length(scores))
  keep <- order(scores, decreasing = TRUE)[seq_len(k)]
  X_sel <- X_cand[keep, , drop = FALSE]

  print(paste0("  kept ", nrow(X_sel), " points",
               " | score range of kept: [",
               round(scores[keep[k]], 4), ", ", round(scores[keep[1]], 4), "]"))

  replay_all <- rbind(replay_all, X_sel)
  labels_all   <- c(labels_all, rep(as.integer(label), nrow(X_sel)))
}

# ---- Write inducing_points.csv (same format as GP_train.R) ----
out_df <- as.data.frame(replay_all)
colnames(out_df) <- paste0("f", seq_len(f) - 1)
out_df$label <- labels_all

out_path <- paste0(args$save_path, "/replay_points.csv")
write.csv(out_df, file = out_path, row.names = FALSE)
print(paste0("GP-sampled replay_points.csv written: ",
             nrow(out_df), " total rows -> ", out_path))
