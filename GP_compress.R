library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(dplyr)
library(ggplot2)
library(stats)
library(optparse)
library(gplite)

set.seed(430)
# Command Line Arguments
option_list <- list(
  make_option(c("--task_id"), type = "numeric", default = 1,
              help = "Task id [default: %default]"),
  make_option(c("--new_digit"), type = "numeric", default = 5,
              help = "New digit to add [default: %default]"),
  make_option(c("-f", "--feature_size"), type = "numeric", default = 64,
              help = "Number of features (X dimension) [default: %default]"),
  make_option(c("--n_tr"), type = "numeric", default = 1500,
              help = "Number of training data *per existing class* [default: %default]"),
  make_option(c("--n_ts"), type = "numeric", default = 3000,
              help = "Number of testing data *per existing class* [default: %default]"),
  make_option(c("--models_dir"), type= "character",
              default = "Result/AE500_Indpoints_withGP_f64(generate100pc-GP-use-Nonfiltered)/task0-5/RData/indpts_Rdata_ntr1500_nts3000_noctr50_f64/",
              help = "Directory containing the GP models [default: %default]"),   
  make_option(c("--data_tr"), type = "character",
              default = "Result/AE500_Indpoints_withGP_f64(generate100pc-GP-use-Nonfiltered)/task0-5/out/f64_task0-5/train_sftmx.csv",
              help = "The path to training dataset"),
  make_option(c("--data_ts"), type = "character",
              default = "Result/AE500_Indpoints_withGP_f64(generate100pc-GP-use-Nonfiltered)/task0-5/out/f64_task0-5/test_sftmx.csv",
              help = "The path to testing dataset")
)

parser <- OptionParser(option_list = option_list)
args <- parse_args(parser)

existingclass_set <- c(0:(args$task_id + 4))  # Existing classes are 0-task_id
GP_c_set <- c(0:(args$task_id + 3))
new_class <- args$new_digit
f <- args$feature_size
n_tr <- args$n_tr
n_ts <- args$n_ts

tau <- 0.90 # threshold used in objective function
pen_scale <- 100  # penalty scale on GP original class (e.g class 4 for GP4)

# Data dictionaries
all_data_X <- list()
all_data_Y <- list()
NNscores_sd <- list()

GPmodels <- list()

# Load GP models 
for (i in GP_c_set) {
  key <- paste0("c", i)
  print(paste0("Loading GP model for class ", key))
  model_path <- paste0(args$models_dir, "GPmodel_", key, ".rda")
  if (file.exists(model_path)) {
    gp_obj <- gp_load(model_path)
    GPmodels[[key]] <- gp_obj
  } else {
    stop(paste("GP model file not found:", model_path))
  }
}

# Load training data (not split into train/val)
train.df <- read.csv(args$data_tr)
for(j in existingclass_set) {
  key <- paste0("c", j)
  num_rows <- nrow(train.df[train.df$label == j, ])
  if(n_tr > num_rows) {
    n_rows <- num_rows
    print(paste0("number of training samples (n_tr) is larger than number of training data for class ", j, 
                 ", load ", n_rows, " samples instead"))
  }
  indices <- sample(num_rows, n_rows, replace = FALSE)
  class_samples <- as.matrix(train.df[train.df$label == j, ])
  class_samples <- class_samples[indices, ]
  all_data_X[[key]] <- class_samples[, 1:f]
  all_data_Y[[key]] <- class_samples[, (f + j + 1), drop = FALSE]

  # calculate sd from nn
  NNscores_sd[[key]] <- sd(all_data_Y[[key]])
}

# Optimal select GPc for new class l and find gamma_cl
optimal_values <- list()
optimal_gammas <- list()

for(c in GP_c_set) {
  key <- paste0("c", c)
  if (is.null(GPmodels[[key]])) {
    stop(paste("GP model for class", c, "not found."))
  }
  gp_model <- GPmodels[[key]]
  GPcc_out <- gp_pred(gp_model, all_data_X[[key]], jitter = 1e-4)
  mean_cc <- GPcc_out$mean
  
  objective_fn <- function(gamma_cl){
    value <- 0
    for(i in GP_c_set){ # existing classes
        pen <- 100
        if(i == c){
          pen <- pen_scale
        }
        X_i <- all_data_X[[paste0("c", i)]]
        GP_out <- gp_pred(gp_model, X_i, jitter = 1e-4)
        mean <- GP_out$mean
        mu <- mean(mean)
        sigma <- sd(mean)
        # find region of prob >= tau
        alpha <- (1 - tau)/2 
        lower_bound <- qnorm(alpha, mean = mu, sd = sigma)
        upper_bound <- qnorm(1-alpha, mean = mu, sd = sigma)
        prob <- pnorm(upper_bound, mean = gamma_cl, sd = NNscores_sd[[paste0("c", new_class)]]) - pnorm(lower_bound, mean = gamma_cl, sd = NNscores_sd[[paste0("c", new_class)]])
        value <- value + pen * prob
    }
    return(value)
  }
  op_result <- optim(
      par = 0.9,
      fn = objective_fn,
      method = "L-BFGS-B",
      lower = 0,
      upper = mean_cc
  )
  
  optimal_values[[paste0("c", c)]] <- op_result$value
  optimal_gammas[[paste0("c", c)]] <- op_result$par

  print(paste0("Optimal gamma_cl for GPc", c, " is: ", op_result$par, 
              " with objective value: ", op_result$value))
}

# Find optimal compress GPc for new class l
optimal_c <- which.min(unlist(optimal_values))
optimal_gamma_cl <- optimal_gammas[[optimal_c]]

print(paste0("Optimal GPc for new class ", new_class, " is: compress GP", optimal_c, 
            " with gamma_cl: ", optimal_gamma_cl))
print("---------------------")


####  Plot, example for GP2 ####
print("Plot an Example - GP2")
X_0 <- all_data_X[["c0"]]
X_1 <- all_data_X[["c1"]]
X_2 <- all_data_X[["c2"]]
X_3 <- all_data_X[["c3"]]
X_4 <- all_data_X[["c4"]] 

gp_model <- GPmodels[["c2"]]
GPc_X0 <- gp_pred(gp_model, X_0, jitter = 1e-4)
GPc_X1 <- gp_pred(gp_model, X_1, jitter = 1e-4)
GPc_X2 <- gp_pred(gp_model, X_2, jitter = 1e-4)
GPc_X3 <- gp_pred(gp_model, X_3, jitter = 1e-4)
GPc_X4 <- gp_pred(gp_model, X_4, jitter = 1e-4)

print(paste0("GPc(X1)  mean of GP$mean =", mean(GPc_X1$mean), ", sd of GP$mean =", sd(GPc_X1$mean)))
print(paste0("GPc(X2) mean of GP$mean =", mean(GPc_X2$mean), ", sd of GP$mean =", sd(GPc_X2$mean)))
print(paste0("NNscores_sd[c5] = ", NNscores_sd[["c5"]]))

x <- seq(-1, 1.1, length.out = 500)
data <- data.frame(
  x = rep(x, 6),
  density = c(dnorm(x, mean = mean(GPc_X0$mean), sd = sd(GPc_X0$mean)), 
              dnorm(x, mean = mean(GPc_X1$mean), sd = sd(GPc_X1$mean)), 
              dnorm(x, mean = mean(GPc_X2$mean), sd = sd(GPc_X2$mean)), 
              dnorm(x, mean = mean(GPc_X3$mean), sd = sd(GPc_X3$mean)), 
              dnorm(x, mean = mean(GPc_X4$mean), sd = sd(GPc_X4$mean)), 
              dnorm(x, mean = 0.220401426801252, sd = NNscores_sd[["c5"]])),
  group = factor(rep(c("class 0", "class 1", "class 2", "class 3", "class 4", "class 5 with mean=0.220401426801252"), each = length(x)))
)

# Plot using ggplot2
plot <- ggplot(data, aes(x = x, y = density, color = group)) +
  geom_line(size = 1) +
  labs(title = "GP2 output distribution on different classes", x = "y-score", y = "Density") +
  theme_minimal() + 
  ylim(0, 0.1)

print(plot)