# GP_visualize_overlap.R
# Visualizes ALL classes' GP-accepted candidate clusters and inducing points
# together in one shared coordinate space, to check whether classes overlap
# (a risk factor for confused replay / catastrophic-forgetting-style errors).
#
# Two complementary views:
#   1. VISUAL: one global, UNSUPERVISED UMAP embedding (fit across all
#      classes' points together, not a separate embedding per class) showing
#      each class's accepted candidate cloud + a 95% coverage ellipse, and
#      each class's inducing points (Z_t). UMAP is used instead of PCA
#      because PCA is linear and label-agnostic — it optimizes for global
#      variance, not class separation, so it can show classes as overlapping
#      even when they're separable via a nonlinear boundary. UMAP is still
#      unsupervised (no label information feeds the embedding), so it's a
#      fair, nonlinear view of the raw geometry — unlike a *supervised* UMAP
#      or LDA, it can't fake separation that isn't there.
#      Note inducing points already deliberately mix in some other-class
#      (Y~=0) points to anchor the decision boundary (see train_GP_v3 /
#      train_GP_laGP in utils.R), so overlap among inducing points across
#      classes is expected by construction — that's why there's no
#      cross-score check on inducing points below, only on candidates.
#   2. QUANTITATIVE: cross-score / cross-accept-rate matrices — class i's
#      accepted candidates (pure per-class mvrnorm draws, NOT mixed with
#      other classes) scored under class j's GP. High off-diagonal values
#      mean class i's replay-eligible material also looks legitimate to
#      class j's GP, i.e. real overlap in the full feature space, not a
#      projection artifact.
#
# Output: <out_path>/GP_visualize_overlap_<classes>.pdf
#         <out_path>/GP_overlap_cross_score_mean_candidates.csv
#         <out_path>/GP_overlap_cross_accept_rate_candidates.csv

library(optparse)
library(gplite)
library(laGP)
library(MASS)
source("utils.R")   # loads ggplot2, stats, scales (via ggplot2), etc.

option_list <- list(
  make_option(c("-f", "--feature_size"), type = "numeric", default = 16,
              help = "Feature dimension (must match GP_train.R) [default: %default]"),
  make_option(c("-p", "--save_path"),    type = "character", default = NULL,
              help = "Directory written by GP_train.R (contains GPparams_*.rds)"),
  make_option(c("--existing_classes"),   type = "character", default = "0,1,2,3,4",
              help = "Comma-separated class labels to compare [default: %default]"),
  make_option(c("--GP_package"),         type = "character", default = "gplite",
              help = "GP package used in GP_train.R: gplite or laGP [default: %default]"),
  make_option(c("--score_threshold"),    type = "numeric",   default = 0.9,
              help = "Acceptance threshold defining each class's 'cluster' [default: %default]"),
  make_option(c("--n_vis"),              type = "numeric",   default = 500,
              help = "Candidate points drawn per class [default: %default]"),
  make_option(c("--data_tr"),            type = "character", default = NULL,
              help = "Optional path to a feature CSV (with a 'label' column) to overlay real data"),
  make_option(c("--n_real"),             type = "numeric",   default = 200,
              help = "Real points per class to overlay, if --data_tr given [default: %default]"),
  make_option(c("--out_path"),           type = "character", default = NULL,
              help = "Directory to write outputs [default: --save_path]"),
  make_option(c("--seed"),               type = "numeric",   default = 42,
              help = "Random seed [default: %default]")
)

parser <- OptionParser(option_list = option_list)
args   <- parse_args(parser)

set.seed(args$seed)

if (is.null(args$save_path)) stop("--save_path is required")
out_path <- if (is.null(args$out_path)) args$save_path else args$out_path
prepare_save_dir(out_path)

f                <- as.integer(args$feature_size)
n_vis            <- as.integer(args$n_vis)
n_real           <- as.integer(args$n_real)
score_threshold  <- args$score_threshold
GP_package       <- args$GP_package
existing_classes <- as.numeric(strsplit(args$existing_classes, ",")[[1]])
class_keys       <- paste0("c", existing_classes)

real_df <- NULL
if (!is.null(args$data_tr)) {
  real_df <- read.csv(args$data_tr)
  real_df$label <- as.numeric(as.character(real_df$label))
}

print(paste0("GP_visualize_overlap: package=", GP_package,
             ", classes=", args$existing_classes,
             ", n_vis=", n_vis,
             ", score_threshold=", score_threshold))

# ---- Load params + GP model for every class ----
params_list <- list()
gp_models   <- list()

for (key in class_keys) {
  params_file <- paste0(args$save_path, "/GPparams_", key, ".rds")
  if (!file.exists(params_file)) stop(paste("Missing params file:", params_file))
  p <- readRDS(params_file)
  if (is.null(p$center) || is.null(p$covariance)) {
    stop(paste0("GPparams_", key, ".rds is missing center/covariance — re-run GP_train.R"))
  }
  params_list[[key]] <- p

  if (GP_package == "gplite") {
    model_file <- paste0(args$save_path, "/GPmodel_", key, ".rda")
    if (!file.exists(model_file)) stop(paste("Missing model file:", model_file))
    gp_models[[key]] <- gp_load(model_file)

  } else if (GP_package == "laGP") {
    da <- darg(list(mle = TRUE), p$Z_t)
    ga <- tryCatch(
      garg(list(mle = TRUE), matrix(p$Y_Z_t)),
      error = function(e) list(start = 1e-3, min = sqrt(.Machine$double.eps), max = 1.0)
    )
    gp_model <- newGPsep(X = p$Z_t, Z = p$Y_Z_t,
                         d = rep(da$start, ncol(p$Z_t)), g = ga$start, dK = TRUE)
    mleGPsep(gp_model, param = "both", tmin = c(da$min, ga$min), tmax = c(da$max, ga$max))
    gp_models[[key]] <- gp_model

  } else {
    stop(paste("Unknown GP_package:", GP_package))
  }
}

score_with <- function(key, X) {
  if (GP_package == "gplite") {
    as.numeric(gp_pred(gp_models[[key]], X, jitter = 1e-4)$mean)
  } else {
    as.numeric(predGPsep(gp_models[[key]], X)$mean)
  }
}

# ---- Draw each class's candidate cloud, score under its OWN GP ----
cand_list <- list()
for (key in class_keys) {
  p <- params_list[[key]]
  covariance <- p$covariance + diag(1e-8, nrow(p$covariance))
  X_cand <- mvrnorm(n_vis, mu = p$center, Sigma = covariance)
  own_score <- score_with(key, X_cand)
  cand_list[[key]] <- list(X = X_cand, score = own_score, accept = own_score >= score_threshold)
}

# ---- Cross-score / cross-accept-rate: class i's accepted candidates scored under class j's GP ----
n_cls <- length(class_keys)
mean_score_cand  <- matrix(NA_real_, n_cls, n_cls, dimnames = list(class_keys, class_keys))
accept_rate_cand <- matrix(NA_real_, n_cls, n_cls, dimnames = list(class_keys, class_keys))

for (ki in class_keys) {
  accepted_X <- cand_list[[ki]]$X[cand_list[[ki]]$accept, , drop = FALSE]
  if (nrow(accepted_X) == 0) {
    print(paste0("  NOTE: ", ki, " had 0/", n_vis,
                 " candidates pass its OWN GP at score_threshold=", score_threshold,
                 " — its cross-score row will be NA (threshold may be too high for this class,",
                 " or its GP is fit weakly; check GP_sample.R's replay output for this class too)"))
    next
  }
  for (kj in class_keys) {
    s_cand <- score_with(kj, accepted_X)
    mean_score_cand[ki, kj]  <- mean(s_cand)
    accept_rate_cand[ki, kj] <- mean(s_cand >= score_threshold)
  }
}

print("Cross-score (mean GP_j score on class_i's ACCEPTED candidates):")
print(round(mean_score_cand, 3))
print("Cross-accept-rate (fraction of class_i's accepted candidates also >= threshold under GP_j):")
print(round(accept_rate_cand, 3))

for (ki in class_keys) {
  for (kj in class_keys) {
    if (ki != kj && !is.na(accept_rate_cand[ki, kj]) && accept_rate_cand[ki, kj] > 0.2) {
      print(paste0("  WARNING: ", ki, "'s accepted candidates are also accepted by ", kj,
                   "'s GP ", round(accept_rate_cand[ki, kj] * 100, 1),
                   "% of the time — possible overlap between ", ki, " and ", kj))
    }
  }
}

write.csv(mean_score_cand,  file = paste0(out_path, "/GP_overlap_cross_score_mean_candidates.csv"))
write.csv(accept_rate_cand, file = paste0(out_path, "/GP_overlap_cross_accept_rate_candidates.csv"))

if (GP_package == "laGP") {
  for (key in class_keys) deleteGPsep(gp_models[[key]])
}

# ---- Build one global PCA over every class's points together ----
combined <- matrix(NA_real_, nrow = 0, ncol = f)
class_col <- character(0)
type_col  <- character(0)

for (key in class_keys) {
  label <- sub("^c", "", key)
  Xc <- cand_list[[key]]$X[cand_list[[key]]$accept, , drop = FALSE]
  combined <- rbind(combined, Xc)
  class_col <- c(class_col, rep(label, nrow(Xc)))
  type_col  <- c(type_col, rep("candidate", nrow(Xc)))

  Zt <- as.matrix(params_list[[key]]$Z_t)
  combined <- rbind(combined, Zt)
  class_col <- c(class_col, rep(label, nrow(Zt)))
  type_col  <- c(type_col, rep("inducing", nrow(Zt)))

  if (!is.null(real_df)) {
    rows <- real_df[real_df$label == as.numeric(label), 1:f, drop = FALSE]
    if (nrow(rows) > 0) {
      n_take <- min(n_real, nrow(rows))
      Xr <- as.matrix(rows[sample(nrow(rows), n_take), , drop = FALSE])
      combined <- rbind(combined, Xr)
      class_col <- c(class_col, rep(label, nrow(Xr)))
      type_col  <- c(type_col, rep("real", nrow(Xr)))
    }
  }
}

umap_config <- umap.defaults
umap_config$random_state <- args$seed
umap_out <- umap(combined, config = umap_config)
proj <- umap_out$layout

plot_df <- data.frame(UMAP1 = proj[, 1], UMAP2 = proj[, 2],
                       class = factor(class_col, levels = as.character(existing_classes)),
                       type = type_col)

palette <- scales::hue_pal()(length(existing_classes))
names(palette) <- as.character(existing_classes)

pdf_path <- paste0(out_path, "/GP_visualize_overlap_", gsub(",", "_", args$existing_classes), ".pdf")
pdf(file = pdf_path, width = 9, height = 7)

# ---- Page 1: global overlay — accepted clusters + 95% ellipses + inducing points ----
p1 <- ggplot()
if (!is.null(real_df)) {
  p1 <- p1 + geom_point(data = subset(plot_df, type == "real"),
                         aes(x = UMAP1, y = UMAP2), color = "grey80", size = 0.8, alpha = 0.4)
}
p1 <- p1 +
  geom_point(data = subset(plot_df, type == "candidate"),
             aes(x = UMAP1, y = UMAP2, fill = class), shape = 21, color = "transparent", size = 1.4, alpha = 0.45) +
  stat_ellipse(data = subset(plot_df, type == "candidate"),
               aes(x = UMAP1, y = UMAP2, color = class), level = 0.95, linewidth = 0.9) +
  geom_point(data = subset(plot_df, type == "inducing"),
             aes(x = UMAP1, y = UMAP2, fill = class), shape = 21, color = "black", size = 2.8, stroke = 0.7) +
  scale_fill_manual(values = palette, name = "class") +
  scale_color_manual(values = palette, guide = "none") +
  labs(
    title = "All classes: GP-accepted clusters & inducing points (shared global UMAP)",
    subtitle = paste0("threshold=", score_threshold, " | filled dots=accepted candidates, ",
                       "ellipse=95% coverage, black-outlined dots=inducing points",
                       if (!is.null(real_df)) " | grey=real data" else "",
                       " | unsupervised UMAP (no label info in the embedding)"),
    x = "UMAP1", y = "UMAP2"
  ) +
  theme_minimal()
print(p1)

# ---- Page 2: inducing points only (cleaner view; expect mixing by design — see header) ----
p2 <- ggplot(subset(plot_df, type == "inducing"), aes(x = UMAP1, y = UMAP2, fill = class)) +
  geom_point(shape = 21, color = "black", size = 3, stroke = 0.8, alpha = 0.85) +
  scale_fill_manual(values = palette, name = "class") +
  labs(title = "Inducing points only, by class (shared global UMAP)",
       subtitle = "each class's inducing set deliberately mixes in other-class points to anchor the boundary, so some mixing here is expected",
       x = "UMAP1", y = "UMAP2") +
  theme_minimal()
print(p2)

# ---- Page 3+4: cross-score / cross-accept-rate heatmaps (candidates only) ----
heatmap_df <- function(mat, value_name) {
  df <- as.data.frame(as.table(mat))
  colnames(df) <- c("source_class", "scored_by_GP", value_name)
  df
}

hm1 <- heatmap_df(mean_score_cand, "mean_score")
p3 <- ggplot(hm1, aes(x = scored_by_GP, y = source_class, fill = mean_score)) +
  geom_tile() +
  geom_text(aes(label = round(mean_score, 2)), size = 3.5) +
  scale_fill_viridis_c(name = "mean\nscore") +
  labs(title = "Cross-score: class i's accepted candidates, scored by class j's GP",
       subtitle = "off-diagonal high values = class i's samples also look real to class j (overlap)",
       x = "scored by GP of class", y = "candidates from class") +
  theme_minimal()
print(p3)

hm2 <- heatmap_df(accept_rate_cand, "accept_rate")
p4 <- ggplot(hm2, aes(x = scored_by_GP, y = source_class, fill = accept_rate)) +
  geom_tile() +
  geom_text(aes(label = paste0(round(accept_rate * 100), "%")), size = 3.5) +
  scale_fill_viridis_c(name = "accept\nrate", limits = c(0, 1)) +
  labs(title = paste0("Cross-accept-rate at threshold=", score_threshold),
       subtitle = "off-diagonal % = how often class i's accepted candidates would ALSO be accepted into class j's replay pool",
       x = "scored by GP of class", y = "candidates from class") +
  theme_minimal()
print(p4)

dev.off()

print(paste0("Saved plots: ", pdf_path))
print(paste0("Saved cross-score CSVs to: ", out_path))
