# GP_visualize.R
# Visualizes, for each class GP trained by GP_train.R:
#   - a cloud of candidate points drawn the same way GP_sample.R draws them
#     (mvrnorm(center, covariance)), colored by GP score
#   - the class's inducing points (Z_t) used to fit/reconstruct the GP
#   - (optional) real class feature rows, for context
#   - the score_threshold acceptance boundary, plus how that threshold compares
#     to the empirical percentiles of the candidate score distribution
#
# Projection: a per-class PCA (center-only, no scaling) fit on
# rbind(candidates, inducing points[, real data]) so the 2D layout reflects
# the actual Euclidean/Mahalanobis-ish geometry of the sampling covariance.
#
# Output: <out_path>/GP_visualize_<classes>.pdf (one scatter + one histogram
# page per class) and a console table of threshold-vs-percentile stats.

library(optparse)
library(gplite)
library(laGP)
library(MASS)
source("utils.R")   # loads ggplot2, stats, etc.

option_list <- list(
  make_option(c("-f", "--feature_size"), type = "numeric", default = 16,
              help = "Feature dimension (must match GP_train.R) [default: %default]"),
  make_option(c("-p", "--save_path"),    type = "character", default = NULL,
              help = "Directory written by GP_train.R (contains GPparams_*.rds)"),
  make_option(c("--existing_classes"),   type = "character", default = "0,1",
              help = "Comma-separated class labels to visualize [default: %default]"),
  make_option(c("--GP_package"),         type = "character", default = "gplite",
              help = "GP package used in GP_train.R: gplite or laGP [default: %default]"),
  make_option(c("--score_threshold"),    type = "numeric",   default = 0.9,
              help = "Acceptance threshold to visualize [default: %default]"),
  make_option(c("--percentiles"),        type = "character", default = "50,75,90,95,99",
              help = "Comma-separated percentiles to compare against score_threshold [default: %default]"),
  make_option(c("--n_vis"),              type = "numeric",   default = 1000,
              help = "Number of candidate points to draw/score for visualization [default: %default]"),
  make_option(c("--data_tr"),            type = "character", default = NULL,
              help = "Optional path to a feature CSV (with a 'label' column) to overlay real class points"),
  make_option(c("--n_real"),             type = "numeric",   default = 300,
              help = "Real points per class to overlay, if --data_tr given [default: %default]"),
  make_option(c("--out_path"),           type = "character", default = NULL,
              help = "Directory to write the PDF [default: --save_path]"),
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
existing_classes <- as.list(as.numeric(strsplit(args$existing_classes, ",")[[1]]))
pctiles          <- as.numeric(strsplit(args$percentiles, ",")[[1]])

real_df <- NULL
if (!is.null(args$data_tr)) {
  real_df <- read.csv(args$data_tr)
  real_df$label <- as.numeric(as.character(real_df$label))
}

print(paste0("GP_visualize: package=", GP_package,
             ", classes=", args$existing_classes,
             ", n_vis=", n_vis,
             ", score_threshold=", score_threshold,
             ", percentiles=", args$percentiles))

pdf_path <- paste0(out_path, "/GP_visualize_", gsub(",", "_", args$existing_classes), ".pdf")
pdf(file = pdf_path, width = 8, height = 6)

summary_rows <- list()

for (j in seq_along(existing_classes)) {
  label <- existing_classes[[j]]
  key   <- paste0("c", label)

  params_file <- paste0(args$save_path, "/GPparams_", key, ".rds")
  if (!file.exists(params_file)) stop(paste("Missing params file:", params_file))
  params <- readRDS(params_file)

  if (is.null(params$center) || is.null(params$covariance)) {
    stop(paste0("GPparams_", key, ".rds is missing center/covariance — re-run GP_train.R"))
  }

  # ---- Load / reconstruct GP model (same as GP_sample.R) ----
  if (GP_package == "gplite") {
    model_file <- paste0(args$save_path, "/GPmodel_", key, ".rda")
    if (!file.exists(model_file)) stop(paste("Missing model file:", model_file))
    gp_model <- gp_load(model_file)

  } else if (GP_package == "laGP") {
    da <- darg(list(mle = TRUE), params$Z_t)
    ga <- tryCatch(
      garg(list(mle = TRUE), matrix(params$Y_Z_t)),
      error = function(e) {
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

  # ---- Draw the same candidate cloud GP_sample.R would draw ----
  center     <- params$center
  covariance <- params$covariance + diag(1e-8, nrow(params$covariance))

  X_cand <- mvrnorm(n_vis, mu = center, Sigma = covariance)
  if (GP_package == "gplite") {
    scores <- as.numeric(gp_pred(gp_model, X_cand, jitter = 1e-4)$mean)
  } else {
    scores <- as.numeric(predGPsep(gp_model, X_cand)$mean)
  }
  if (GP_package == "laGP") deleteGPsep(gp_model)

  accept <- scores >= score_threshold

  # ---- Threshold vs. percentile comparison ----
  thresh_rank_pct <- mean(scores < score_threshold) * 100  # % of candidates below threshold
  q_vals <- quantile(scores, probs = pctiles / 100)

  print(paste0("Class ", label, ": score_threshold=", score_threshold,
               " sits at the ", round(thresh_rank_pct, 1),
               "th percentile of the candidate score distribution",
               " (accept rate ", round(mean(accept) * 100, 1), "%)"))
  for (k in seq_along(pctiles)) {
    print(paste0("  P", pctiles[k], " score cutoff = ", round(q_vals[k], 4)))
  }

  summary_rows[[key]] <- data.frame(
    label = label,
    score_threshold = score_threshold,
    threshold_percentile = thresh_rank_pct,
    accept_rate_pct = mean(accept) * 100,
    setNames(as.list(round(q_vals, 4)), paste0("P", pctiles))
  )

  # ---- Real data overlay (optional) ----
  real_X <- NULL
  if (!is.null(real_df)) {
    rows <- real_df[real_df$label == label, 1:f, drop = FALSE]
    if (nrow(rows) > 0) {
      n_take <- min(n_real, nrow(rows))
      real_X <- as.matrix(rows[sample(nrow(rows), n_take), , drop = FALSE])
    }
  }

  Z_t <- as.matrix(params$Z_t)

  # ---- Local PCA (center-only) shared by candidates/inducing/real ----
  combined <- rbind(X_cand, Z_t)
  types    <- c(rep("candidate", nrow(X_cand)), rep("inducing", nrow(Z_t)))
  if (!is.null(real_X)) {
    combined <- rbind(combined, real_X)
    types    <- c(types, rep("real", nrow(real_X)))
  }
  pca  <- prcomp(combined, center = TRUE, scale. = FALSE)
  proj <- pca$x[, 1:2]
  var_explained <- round(100 * (pca$sdev[1:2]^2) / sum(pca$sdev^2), 1)

  plot_df <- data.frame(PC1 = proj[, 1], PC2 = proj[, 2], type = types)
  plot_df$score  <- NA_real_
  plot_df$accept <- NA
  plot_df$score[types == "candidate"]  <- scores
  plot_df$accept[types == "candidate"] <- accept

  # fill color stays on one continuous scale spanning the true score range,
  # so accepted points (often clustered near/above 1.0) aren't clipped/flattened;
  # accept/reject is instead shown via border color + point size below
  score_range <- range(scores)
  color_breaks <- sort(unique(round(c(score_range, score_threshold), 3)))

  p <- ggplot()
  if (!is.null(real_X)) {
    p <- p + geom_point(data = subset(plot_df, type == "real"),
                         aes(x = PC1, y = PC2), color = "grey70", size = 1, alpha = 0.5)
  }
  p <- p +
    geom_point(data = subset(plot_df, type == "candidate"),
               aes(x = PC1, y = PC2, fill = score, color = accept, size = accept),
               shape = 21, stroke = 0.6, alpha = 0.85) +
    scale_fill_viridis_c(name = "GP score", limits = score_range, breaks = color_breaks) +
    scale_color_manual(values = c(`TRUE` = "black", `FALSE` = "grey85"), guide = "none") +
    scale_size_manual(values = c(`TRUE` = 2.4, `FALSE` = 1.3), guide = "none") +
    geom_point(data = subset(plot_df, type == "inducing"),
               aes(x = PC1, y = PC2), shape = 4, size = 3, stroke = 1.2, color = "red") +
    labs(
      title = paste0("Class ", label, ": candidate cloud colored by GP score"),
      subtitle = paste0("n_cand=", n_vis, " | accept@", score_threshold, "=",
                         round(mean(accept) * 100, 1), "% (=P", round(thresh_rank_pct, 1), ") | ",
                         "grey=real data, red X=inducing pts, black-outlined dots=accepted"),
      x = paste0("PC1 (", var_explained[1], "%)"),
      y = paste0("PC2 (", var_explained[2], "%)")
    ) +
    theme_minimal()
  print(p)

  # ---- Score histogram with threshold + percentile lines ----
  hist_df <- data.frame(score = scores)
  vline_df <- data.frame(
    value = c(score_threshold, unname(q_vals)),
    label = c(paste0("threshold=", score_threshold), paste0("P", pctiles)),
    kind  = c("threshold", rep("percentile", length(q_vals)))
  )
  h <- ggplot(hist_df, aes(x = score)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    geom_vline(data = vline_df, aes(xintercept = value, linetype = kind, color = kind), linewidth = 0.7) +
    geom_text(data = vline_df, aes(x = value, y = Inf, label = label, color = kind),
              angle = 90, hjust = 1.1, vjust = -0.3, size = 3, show.legend = FALSE) +
    scale_linetype_manual(values = c(threshold = "solid", percentile = "dashed")) +
    scale_color_manual(values = c(threshold = "red", percentile = "black")) +
    labs(title = paste0("Class ", label, ": GP score distribution over candidates"),
         subtitle = "solid red = score_threshold, dashed = percentile cutoffs",
         x = "GP score", y = "count") +
    theme_minimal()
  print(h)
}

dev.off()

summary_df <- do.call(rbind, summary_rows)
summary_csv <- paste0(out_path, "/GP_visualize_threshold_vs_percentile.csv")
write.csv(summary_df, file = summary_csv, row.names = FALSE)

print(paste0("Saved plots: ", pdf_path))
print(paste0("Saved threshold-vs-percentile summary: ", summary_csv))
