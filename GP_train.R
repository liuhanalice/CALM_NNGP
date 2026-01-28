library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(dplyr)
library(ggplot2)
library(stats)
library(optparse)
library(gplite)
library(laGP)
source("utils.R")

set.seed(430)
option_list <- list(
  make_option(c("--n_tr"), type = "numeric", default = 1000,
              help = "Number of training data *per existing class* [default: %default]"),
  make_option(c("--n_ts"), type = "numeric", default = 1500,
              help = "Number of testing data *per existing class* [default: %default]"),
  make_option(c("--n_octr"), type = "numeric", default = 50,
              help = "Number of training data from other classes (in total) 
              to include in training target GP model  [default: %default]"),
  make_option(c("--val_fac"), type = "numeric", default = 0.8,
              help = "The train/validation factor [default: %default]"),
  make_option(c("-f", "--feature_size"), type = "numeric", default = 16,
              help = "Number of features (X dimension) [default: %default]"),
  make_option(c("-p", "--save_path"), type = "character", default = NULL,
              help = "The directory to save files"),
  make_option(c("--data_tr"), type = "character",
              default = "runs_mnist_continual/run_20250918_213623/task0/train_feat.csv",
              help = "The path to training dataset"),
  make_option(c("--data_ts"), type = "character",
              default = "runs_mnist_continual/run_20250918_213623/task0/test_feat.csv",
              help = "The path to testing dataset"),
  make_option(c("--n_indcpts"), type = "numeric", default = 1000,
              help = "Number of inducing points [default: %default]"),
  make_option(c("--last_class"), type = "numeric", default = 4,
              help = "Existing classes are 0-last_class, train GP models for digit 0-last_class [default: %default]"),
  make_option(c("--use_Y_logspace"), type = "logical", default = FALSE,
            help = "Whether to apply log-space transform to Y values (TRUE or FALSE) [default: %default]"),
  make_option(c("--GP_package"), type = "character", default = "gplite",
              help = "The package to use for GP training (gplite or laGP) [default: %default]")
)
parser <- OptionParser(option_list = option_list)
args <- parse_args(parser)

f <- args$feature_size
other_class_sample_num <- args$n_octr
num_inducing <- args$n_indcpts
GP_package <- args$GP_package

# print(paste0("DEBUG: ", "n_tr=", args$n_tr, ", n_ts=", args$n_ts, ", n_octr=", args$n_octr, ", n_indcpts=", args$n_indcpts, ", f_size=", args$feature_size))

# Create save path
if (is.null(args$save_path)){
  args$save_path <- paste0(args$save_path,"/indpts_Rdata_ntr", 
                           toString(args$n_tr), "_nts",
                           toString(args$n_ts), "_noctr",
                           toString(args$n_octr), "_f",
                           toString(args$feature_size))
}
prepare_save_dir(save_path=args$save_path)
pdf(file = paste0(args$save_path, "/Rplots_train.pdf"))

# Data dictionaries
all_data_X <- list()
all_data_Y <- list()
val_data_X <- list()
val_data_Y <- list()
test_data_X <- list()
test_data_Y <- list()
test_data_label <- list() # for convenient


GPmodel_train <- list()
GPresult_train <- list()
mse_train <- list()

is_test <- FALSE

existingclass_set <- as.list(0:args$last_class)

if (is_test) {
  load(paste0(args$save_path, "/GPmodel_train.RData"))
} else {
  #########  Train GP for existing classes #########
  
  # Load data
  df <- read.csv(args$data_tr)
  
  # Separate train and val 
  datasets <- split_train_val(df, "label", args$val_fac)
  train.df <- datasets$train.df
  val.df <- datasets$val.df

  # Use log-space (log1p) transform for Y values if specified
  if (args$use_Y_logspace) {
  #   train.df[, (f + 1): (ncol(train.df)-1)] <- log1p(train.df[, (f + 1): (ncol(train.df)-1)])
  #   val.df[, (f + 1): (ncol(val.df)-1)] <- log1p(val.df[, (f + 1): (ncol(val.df)-1)])
    train.df[, (f + 1): (ncol(train.df)-1)] <- exp(train.df[, (f + 1): (ncol(train.df)-1)] + 1)
    val.df[, (f + 1): (ncol(val.df)-1)] <- exp(val.df[, (f + 1): (ncol(val.df)-1)] + 1)
    print("Using log-space (exp y) transform for Y values")
  } else {
    print("Using original Y values (no log-space transform)")
  }
  print(paste0("Using GP package: ", GP_package))
  print("======================================")

  # Init inducing points and labels matrices
  inducing_points <- matrix(NA, nrow = 0, ncol = f)
  inducing_points_GTlabels <- matrix(NA, nrow = 0, ncol = 1)

  # Train GP models for each existing class
  print("---- Start Training GPs for existing classes ----")
  for (j in seq_along(existingclass_set)) { # j = 1,2,3,4, ...
    label <- existingclass_set[[j]] # label = 0,1,2,3, ...
    key <- paste0("c", label)

    sampled_other <- load_classes_other_than_label(train.df, label, existingclass_set, other_class_sample_num)
    sampled_other_val <- load_classes_other_than_label(val.df, label, existingclass_set, other_class_sample_num)
    
    current_data <- load_data_per_class(train.df, label, args$n_tr)
    current_data_val <- load_data_per_class(val.df, label, args$n_tr)


    all_data_X[[key]] <- current_data[, 1:f]
    all_data_Y[[key]] <- current_data[, (f + j), drop = FALSE]
    
    val_data_X[[key]] <- current_data_val[, 1:f]
    val_data_Y[[key]] <- current_data_val[, (f + j), drop = FALSE]

    X <- rbind(all_data_X[[key]], sampled_other[, 1:f])
    Y <- rbind(all_data_Y[[key]], sampled_other[, (f + j), drop = FALSE])
    val.X <- rbind(val_data_X[[key]], sampled_other_val[, 1:f])
    val.Y <- rbind(val_data_Y[[key]], sampled_other_val[, (f + j), drop = FALSE])

    print(paste0("class ", label, " training sample size (indclude ", nrow(sampled_other)," samples from other classes): ", nrow(X)))
    print(paste0("class ", label, " validation set sample size (indclude ", nrow(sampled_other_val)," samples from other classes): ", nrow(val.X)))
  
    ### train_GP: bind target class and other classes ###
    # GPj <- train_GP(X=X, Y=Y, val.X=val.X, val.Y=val.Y, label=label, use_inducing = TRUE, num_inducing = num_inducing)
    if (GP_package == 'gplite') {
      GPj <- train_GP_v3(X_t=all_data_X[[key]], X_otc=sampled_other[, 1:f], Y_t=all_data_Y[[key]], Y_otc=sampled_other[, (f + j), drop = FALSE],
                        val.X=val.X, val.Y=val.Y,
                        label=label, use_inducing = TRUE, num_inducing = num_inducing)
    } else if (GP_package == 'laGP') {
      GPj <- train_GP_laGP(X_t=all_data_X[[key]], X_otc=sampled_other[, 1:f], Y_t=all_data_Y[[key]], Y_otc=sampled_other[, (f + j), drop = FALSE],
                        val.X=val.X, val.Y=val.Y,
                        label=label, use_inducing = TRUE, num_inducing = num_inducing)
    }

    GPmodel_train[[key]] <- GPj$GPmodel
    GPresult_train[[key]] <- GPj$GPresult
    mse_train[[key]] <- GPj$mse
    inducing_points <- rbind(inducing_points, GPj$inducing_points)
    # assuming inducing points are all from the same class j [original points as inducing points]
    inducing_points_GTlabels <- rbind(inducing_points_GTlabels, matrix(label, nrow = nrow(GPj$inducing_points), ncol = 1)) 

    print(paste0("Current inducing_points dim: ", dim(inducing_points)[1], " x ", dim(inducing_points)[2]))
    print(GPj$plot)
    gp_save(GPj$GPmodel, paste0(args$save_path, "/GPmodel_", key, ".rda"))
  }
  print("Train Finished")
  print("---------------------")

  # Save inducing points
  # write.csv(inducing_points, file=paste0(args$save_path, "/inducing_points.csv"), row.names = FALSE)
  ## Inducing points Label assignment ##
  # option 1: build pred matrix
  # indcpts_pred_mat <- build_inducing_pts_pred_matrix(inducing_points, GPmodel_train)
  # option 2: save GT labels)
  indcpts_pred_mat <- data.frame(inducing_points, label = inducing_points_GTlabels)
  write.csv(indcpts_pred_mat, file=paste0(args$save_path, "/inducing_points.csv"), row.names = FALSE)
  print("Inducing Points Saved")
  print("---------------------")
}

#########  Test GP for existing classes #########
test.df <- read.csv(args$data_ts)
n_ts <- (args$n_ts) * (args$last_class + 1)
  print(paste0("Expected testing data total: ", n_ts))
if (n_ts > nrow(test.df)) {
    print(paste0("n_ts is greater than the number of rows in the data frame, using all rows:", nrow(test.df)))
    n_ts <- nrow(test.df)
}
test.df <- test.df[1:n_ts, ]
print(paste0("total test df size: ", nrow(test.df)))

str(test.df$label)
table(test.df$label)
test.df$label <- as.numeric(as.character(test.df$label))
for (j in seq_along(existingclass_set)) {
  # label <- j - 1
  label <- existingclass_set[[j]]
  key <- paste0("c", label)
  test_df <- test.df[test.df$label == label, ]
  test_data_X[[key]] <- as.matrix(test_df[, 1:f])
  test_data_Y[[key]] <- as.matrix(test_df[, (f + label + 1), drop = FALSE])
  test_data_label[[key]] <- matrix(label, nrow = nrow(test_data_X[[key]]), ncol=1)
  print(paste0("class ", label, " test data size:", nrow(test_df)))
}

testsets <- do.call(rbind, lapply(c(paste0("c", existingclass_set)), function(index) test_data_X[[index]]))
testsets_labels <- do.call(rbind, lapply(c(paste0("c", existingclass_set)), function(index) test_data_label[[index]]))

print(paste0("Testset GPs:"))
test_result <- test_GPs(GPmodels=GPmodel_train, test_classes=existingclass_set, GPs=existingclass_set, test_data=testsets, test_data_label=testsets_labels, gp_package = GP_package)
print_GP_distributions_stats(test_result$GP_test_mean_mat_with_label)
test_result_plots <- plot_GP_distributions(test_result$GP_test_mean_mat_with_label, 
                    normal_plot_title="output distribution on different classes (testset)", normal_ymin=0, normal_ymax=0.1,
                    histogram_plot_title="histogram of predGPsep scores by class (testset)", histogram_ymin=0, histogram_ymax=600)

#########  Save Structured Result #########
save(file = paste0(args$save_path, "/GPmodel_train.RData"), GPmodel_train, GPresult_train, mse_train, train.df, val.df, all_data_X, all_data_Y, val_data_X, val_data_Y, test_data_X, test_data_Y, test_data_label)

print("Save Results")
dev.off()

