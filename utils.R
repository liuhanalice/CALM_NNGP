library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(dplyr)
library(ggplot2)
library(stats)
library(gplite)
library(laGP)

# Select num_rows of from matrix mat randomly
random_select_rows <- function(mat, num_rows) {
  if (!is.matrix(mat)) stop("Input must be a matrix")
  if (num_rows > nrow(mat)) {
    print("use all rows")
    num_rows = nrow(mat)
  }
  selected_rows <- sample(nrow(mat), num_rows)
  return(mat[selected_rows, , drop = FALSE])
}

# Prepare saving directory
prepare_save_dir <- function(save_path){
  if (!dir.exists(save_path)) {
    if (dir.create(save_path, recursive = TRUE)) {
      cat("Directory created successfully:", save_path, "\n")
    } else {
      cat("Failed to create the directory:", save_path, "\n")
    }
  } else {
    cat("Directory already exists:", save_path, "\n")
  }
}

# Split train and validation data per class by "label_col" column, returns train and val dataframes
split_train_val <- function(df, label_col, val_fac = 0.8) {
  if (val_fac <= 0 || val_fac >= 1) {
    stop("val_fac must be between 0 and 1")
  }
  idx_train <- unlist(lapply(split(1:nrow(df), df[[label_col]]), # split by label
                      function(class_idx) {
                        sample(class_idx, size = floor(length(class_idx) * val_fac))
                      }))
  
  train.df <- df[idx_train, ]
  val.df <- df[-idx_train, ]
  return(list(train.df = train.df, val.df = val.df))
}

# Load data for class = label with size = n
load_data_per_class <- function(df, label, n){
   if (! is.data.frame(df)) {
    stop("Input must be a data frame")
  }

  label_df <- as.matrix(df[df[, "label"] == label, ])
  
  if (n > nrow(label_df)) {
    print(paste0("n is greater than the number of rows in the data frame, using all rows:", nrow(label_df)))
    n <- nrow(label_df)
  }
  select_index <- sample(nrow(label_df), n, replace = FALSE)
  result_df <- label_df[select_index, ]
  return(result_df)
}

# Load classes (total) of size n from classes othere than label
load_classes_other_than_label <- function(df, label, existingclass_set, n){
  other_class <- as.matrix(df[df[, "label"] %in% setdiff(existingclass_set, label), ])
  if (n > nrow(other_class)) {
    print(paste0("n is greater than the number of rows in the data frame, using all rows:", nrow(other_class)))
    n <- nrow(other_class)
  }
  sampled_other <- random_select_rows(other_class, n)
  
  return (sampled_other)
}

# Train GP model
train_GP <- function(X, Y, val.X, val.Y, label, use_inducing, num_inducing) {
  if (! is.matrix(X) || ! is.matrix(Y) || ! is.matrix(val.X) || ! is.matrix(val.Y)) {
    stop("X, Y, val.X, and val.Y must be matrices")
  }
  if (nrow(X) != nrow(Y)) {
    stop("Number of rows in X and Y must be equal")
  }
  if (nrow(val.X) != nrow(val.Y)) {
    stop("Number of rows in val.X and val.Y must be equal")
  }

  # train GP
  if (use_inducing){
    if(num_inducing > nrow(X)){
      num_inducing <- nrow(X)
      print(paste0("Selecting ", num_inducing, " inducing points"))
    }
    # use manually selected inducing points
    set.seed(42)
    Z <- X[sample(1:nrow(X), num_inducing), , drop = FALSE]

    gp_model <- gp_init(cf_sexp(), method = method_fitc(inducing = Z))
  }
  else{
    gp_model <- gp_init(cf_sexp(), lik = lik_gaussian(), method = method_full())
  }
 
  # gp_model <- gp_optim(gp_model, X, Y, verbose = FALSE, restarts = 3, tol_param = 0.05, maxiter = 1000)
  gp_model <- gp_optim(gp_model, X, Y, verbose = FALSE, restarts = 3, tol_param = 0.05, maxiter = 1000, jitter = 1e-3)

  out <- gp_pred(gp_model, X, jitter = 1e-4)
  out.val <- gp_pred(gp_model, val.X, jitter = 1e-4)
  if (use_inducing) {
    inducing_points <- gp_model$method$inducing
  }
  else {
    if(num_inducing < nrow(X)){
      inducing_points <- X[sample(nrow(X), num_inducing), , drop = FALSE]
    }else{
      inducing_points <- X
    }
  }
  
  mse <- norm(out.val$mean - val.Y, "2")
  print(paste0("validation MSE for class", label, ": ", mse))


  # train vs. val MSE plot
  plot_df <- data.frame(Y_true = as.numeric(val.Y), Y_pred = out.val$mean)
  plot <- ggplot(plot_df, aes(x = Y_true, y = Y_pred)) +
    geom_point() +
    ggtitle(paste0("Validation True vs. Pred (class=", label, ")")) +
    xlab("Y_true") + ylab("Y_pred")

  return(list(GPmodel = gp_model, GPresult = out, mse = mse, plot = plot, inducing_points = inducing_points))
}

# Train GP version 2
train_GP_v2 <- function(X_t, X_otc, Y_t, Y_otc, val.X, val.Y, label, use_inducing, num_inducing) {
  if (nrow(val.X) != nrow(val.Y)) {
    stop("Number of rows in val.X and val.Y must be equal")
  }
  # X_otc len = 50
  X <- rbind(X_t, X_otc)
  Y <- rbind(Y_t, Y_otc)
  if (! is.matrix(X) || ! is.matrix(Y)) {
    stop("X and Y must be matrices")
  }
  if (nrow(X) != nrow(Y)) {
    stop("Number of rows in X and Y must be equal")
  }
  
  num_total_points <- nrow(X)
  if (use_inducing){
    if(num_inducing > num_total_points){
      num_inducing <- num_total_points
      print(paste0("Selecting ", num_inducing, " inducing points"))
    }
    # use manually selected inducing points(randomly selected)
    set.seed(42)
    # num_inducing as total points select for this GP model (X_t + X_otc)
    num_inducing_t = floor(num_inducing * nrow(X_t)/num_total_points)
    num_inducing_otc = num_inducing - num_inducing_t
    Z_t <- X_t[sample(1:nrow(X_t), num_inducing_t), , drop = FALSE]
   
    Z_otc <- X_otc[sample(1:nrow(X_otc), num_inducing_otc), , drop = FALSE]
    Z <- rbind(Z_t, Z_otc)
    gp_model <- gp_init(cf_sexp(), method = method_fitc(inducing = Z))
  }
  else{
    gp_model <- gp_init(cf_sexp(), lik = lik_gaussian(), method = method_full())
  }

  gp_model <- gp_optim(gp_model, X, Y, verbose = FALSE, restarts = 3, tol = 1e-04, tol_param = 0.1, maxiter = 500, jitter = 1e-3)
  out <- gp_pred(gp_model, X, jitter = 1e-4)
  out.val <- gp_pred(gp_model, val.X, jitter = 1e-4)
  
  # save only target class inducing points (for manually selected methods)
  if (use_inducing) {
    # inducing_points <- gp_model$method$inducing
    inducing_points <- Z_t
  }
  else { # NOT USED ANYMORE
    if(num_inducing_t < nrow(X_t)){
      inducing_points <- X_t[sample(nrow(X_t), num_inducing_t), , drop = FALSE]
    }else{
      inducing_points <- X_t
    }
  }
  mse <- norm(out.val$mean - val.Y, "2")
  print(paste0("validation MSE for class", label, ": ", mse))


  # train vs. val MSE plot
  plot_df <- data.frame(Y_true = as.numeric(val.Y), Y_pred = out.val$mean)
  plot <- ggplot(plot_df, aes(x = Y_true, y = Y_pred)) +
    geom_point() +
    ggtitle(paste0("Validation True vs. Pred (class=", label, ")")) +
    xlab("Y_true") + ylab("Y_pred")

  return(list(GPmodel = gp_model, GPresult = out, mse = mse, plot = plot, inducing_points = inducing_points))
}


# Train GP version 3: num_inducing are inducing points for **target class**
train_GP_v3 <- function(X_t, X_otc, Y_t, Y_otc, val.X, val.Y, label, use_inducing, num_inducing) {
  if (nrow(val.X) != nrow(val.Y)) {
    stop("Number of rows in val.X and val.Y must be equal")
  }
  # X_otc len = 50
  X <- rbind(X_t, X_otc)
  Y <- rbind(Y_t, Y_otc)
  # if(label == 2){
  #   print("Debug")
  #   print(paste0("X: ", X))
  #   print(paste0("Y: ", Y))
    
  # }
  if (! is.matrix(X) || ! is.matrix(Y)) {
    stop("X and Y must be matrices")
  }
  if (nrow(X) != nrow(Y)) {
    stop("Number of rows in X and Y must be equal")
  }
  
  num_total_points <- nrow(X)
  if (use_inducing){
    if(num_inducing > nrow(X_t)){
      num_inducing <- nrow(X_t)
      print(paste0("Selecting ", num_inducing, " inducing points"))
    }
    # use manually selected inducing points(randomly selected)
    set.seed(42)
    # num_inducing only for target class (X_t)
    Z_t <- X_t[sample(1:nrow(X_t), num_inducing), , drop = FALSE]

    gp_model <- gp_init(cf_sexp(), method = method_fitc(inducing = Z_t))
    # gp_model <- gp_init(cf_sexp(), lik = lik_gaussian(), method = method_full()) # TODO: test not training with inducing points
  }
  else{
    gp_model <- gp_init(cf_sexp(), lik = lik_gaussian(), method = method_full())
  }

  gp_model <- gp_optim(gp_model, X, Y, verbose = FALSE, restarts = 3, tol = 1e-04, tol_param = 0.1, maxiter = 500, jitter = 1e-6)
  out <- gp_pred(gp_model, X, jitter = 1e-6)
  out.val <- gp_pred(gp_model, val.X, jitter = 1e-6)
  
  # save only target class inducing points (for manually selected methods)
  if (use_inducing) {
    # inducing_points <- gp_model$method$inducing
    inducing_points <- Z_t
  }
  else { # return all training points
    inducing_points <- X_t
  }
  mse <- norm(out.val$mean - val.Y, "2")
  print(paste0("validation MSE for class", label, ": ", mse))


  # train vs. val MSE plot
  plot_df <- data.frame(Y_true = as.numeric(val.Y), Y_pred = out.val$mean)
  plot <- ggplot(plot_df, aes(x = Y_true, y = Y_pred)) +
    geom_point() +
    ggtitle(paste0("Validation True vs. Pred (class=", label, ")")) +
    xlab("Y_true") + ylab("Y_pred")

  return(list(GPmodel = gp_model, GPresult = out, mse = mse, plot = plot, inducing_points = inducing_points))
}


# Train GP version 3: num_inducing are inducing points for **target class**
train_GP_laGP <- function(X_t, X_otc, Y_t, Y_otc, val.X, val.Y, label, use_inducing, num_inducing) {
  if (nrow(val.X) != nrow(val.Y)) {
    stop("Number of rows in val.X and val.Y must be equal")
  }
  # X_otc len = 50
  X <- rbind(X_t, X_otc)
  Y <- rbind(Y_t, Y_otc)

  if (! is.matrix(X) || ! is.matrix(Y)) {
    stop("X and Y must be matrices")
  }
  if (nrow(X) != nrow(Y)) {
    stop("Number of rows in X and Y must be equal")
  }
  
  # laGP, not using inducing points, but sample some points to save as "inducing/sampling reference points"
  gp_model <- newGPsep(X=X, Z=Y, d = rep(10, ncol(X)), g = 1e-6, dK = TRUE)
  mleGPsep(gp_model, param = "both") # tmin and tmax by default, see laGP reference
  
  out <- predGPsep(gp_model, X)
  out.val <- predGPsep(gp_model, val.X)

  num_total_points <- nrow(X)
  
  if (use_inducing){
    if(num_inducing > nrow(X_t)){
      num_inducing <- nrow(X_t)
      print(paste0("Selecting ", num_inducing, " inducing points"))
    }
    # use manually selected inducing points(randomly selected)
    set.seed(42)
    # num_inducing only for target class (X_t)
    Z_t <- X_t[sample(1:nrow(X_t), num_inducing), , drop = FALSE]

     inducing_points <- Z_t
  }
  else { # return all training points
    inducing_points <- X_t
  }

  mse <- norm(out.val$mean - val.Y, "2")
  print(paste0("validation MSE for class", label, ": ", mse))

  # train vs. val MSE plot
  plot_df <- data.frame(Y_true = as.numeric(val.Y), Y_pred = out.val$mean)
  plot <- ggplot(plot_df, aes(x = Y_true, y = Y_pred)) +
    geom_point() +
    ggtitle(paste0("Validation True vs. Pred (class=", label, ")")) +
    xlab("Y_true") + ylab("Y_pred")

  return(list(GPmodel = gp_model, GPresult = out, mse = mse, plot = plot, inducing_points = inducing_points))
}



# Test GP models
test_GPs <- function(GPmodels, test_classes, GPs, test_data, test_data_label, gp_package = 'gplite') {
  if (! is.list(GPmodels)) {
    stop("GPmodels must be a list (dictionaries)")
  }
  if (! is.matrix(test_data) || ! is.matrix(test_data_label)) {
    stop("test_data and test_data_label must be matrices")
  }

  GP_test_mean_mat <- matrix(NA_real_, nrow = nrow(test_data), ncol = length(GPs)) # rows are test data (corresponds to test_data_label), columns are GPj prediction value
  for (j in seq_along(GPs)) {
    label <- GPs[[j]]
    key <- paste0("c", label)

    if (!key %in% names(GPmodels)) {
      stop(paste0("GP model for class ", label, " not found"))
    }

    if (gp_package == 'gplite') {
      out.test <- gp_pred(GPmodels[[key]], test_data, jitter = 1e-4)
    } else if (gp_package == 'laGP') {
      out.test <- predGPsep(GPmodels[[key]], test_data)
    } else {
      stop("Unsupported GP package. Use 'gplite' or 'laGP'.")
    }

    GP_test_mean_mat[, j] <- out.test$mean
  }
  colnames(GP_test_mean_mat) <- paste0("c", GPs)

  predicted_classes <- sapply(1:nrow(GP_test_mean_mat), function(i) {
    which.max(GP_test_mean_mat[i, ]) - 1
  })
  predicted_classes <- as.numeric(predicted_classes)
  true_labels <- as.numeric(test_data_label[, 1])

  # total accuracy
  test_accuracy <- sum(predicted_classes == true_labels) / nrow(test_data_label)
  # print(paste0("DEBUG: sum = ", sum(predicted_classes == test_data_label[, 1])))
  # print(paste0("DEBUG: total test rows = ", nrow(test_data_label)))

  print(paste0("Total accuracy: ", test_accuracy))
  
  # confusion matrix per class
  predicted_labels <- predicted_classes
  for (label in  unlist(test_classes)) {
    positive_true <- true_labels == label
    positive_pred <- predicted_labels == label
    
    TP <- sum(positive_true & positive_pred)
    FP <- sum(!positive_true & positive_pred)
    FN <- sum(positive_true & !positive_pred)
    TN <- sum(!positive_true & !positive_pred)
    
    print(paste0("Confusion Matrix for Class:", label))
    print(data.frame(
      Actual_Positive = c(TP, FN),
      Actual_Negative = c(FP, TN),
      row.names = c("Predicted_Positive", "Predicted_Negative")
    ))

    print(paste0("Accuracy: ", (TP + TN) / (TP + FP + TN + FN)))
    print(paste0("Precision: ", TP / (TP + FP)))
    print(paste0("Recall for Class: ", TP / (TP + FN)))
    print(paste0("F1 Score for Class: ", 2 * TP / (2 * TP + FP + FN)))
  }

  GP_test_mean_mat_with_label <- cbind(GP_test_mean_mat, label = test_data_label[, 1])
  return(list(GP_test_mean_mat_with_label = GP_test_mean_mat_with_label))
}

# Print GP distributions statistics
print_GP_distributions_stats <- function(GP_mean_mat_with_label) {
  if (! is.matrix(GP_mean_mat_with_label)) {
    stop("Input must be a metrix with GP predictions and a 'label' column")
  }

  gp_model_names <- colnames(GP_mean_mat_with_label)[colnames(GP_mean_mat_with_label) != "label"]
  true_labels <- GP_mean_mat_with_label[, "label"]
  class_labels <- sort(unique(true_labels))

  for (gp_name in gp_model_names) {
    test_preds <- GP_mean_mat_with_label[, gp_name]

    means <- sapply(class_labels, function(label) {
      mean(test_preds[true_labels == label])
    })

    sds <- sapply(class_labels, function(label) {
      sd(test_preds[true_labels == label])
    })
    print(paste0("GP", gp_name, " distribution statistics for", paste(class_labels, collapse = ", "), ":"))
    print(paste0("mean: ", paste(unname(means), collapse = ", ")))
    print(paste0("sd: ", paste(unname(sds), collapse = ", ")))
    print("---------------------")
  }
}

# Plot distribution of GP predictions as normal distribution
plot_GP_distributions <- function(GP_mean_mat_with_label, normal_plot_title, normal_ymin=0, normal_ymax=0.1, histogram_plot_title, histogram_ymin=0, histogram_ymax=600) {
  if (! is.matrix(GP_mean_mat_with_label)) {
    stop("Input must be a metrix with GP predictions and a 'label' column")
  }

  x <- seq(-1, 2, length.out = 500)
  normal_plots <- list()
  histogram_plots <- list()

  gp_model_names <- colnames(GP_mean_mat_with_label)[colnames(GP_mean_mat_with_label) != "label"]
  true_labels <- GP_mean_mat_with_label[, "label"]
  class_labels <- sort(unique(true_labels))

  for (gp_name in gp_model_names) {
    test_preds <- GP_mean_mat_with_label[, gp_name]

    density_data <- unlist(lapply(class_labels, function(label) {
      idx <- which(true_labels == label)
      mu <- mean(test_preds[idx])
      sd <- sd(test_preds[idx])
      dnorm(x, mean = mu, sd = sd)
    }))
    # as normal distribution
    df <- data.frame(
      x = rep(x, times = length(class_labels)),
      density = density_data,
      group = factor(rep(paste0("class", class_labels), each = length(x)))
    )
    p <- ggplot(df, aes(x = x, y = density, color = group)) +
      geom_line(linewidth = 1) +
      labs(title = paste0("GP", gp_name, " ", normal_plot_title), x = "y-score", y = "Density") +
      theme_minimal() +
      ylim(normal_ymin, normal_ymax)
    print(p)
    normal_plots[[gp_name]] <- p
    # histogram
    hist_df <-  data.frame(
      score = test_preds,
      class = factor(true_labels)
    )
    histogram <- ggplot(hist_df, aes(x = score, fill = class)) +
      geom_histogram(binwidth = 0.01, alpha = 0.7, position = "identity") +
      labs(title = paste0("GP", gp_name, " ", histogram_plot_title), x = "y-score", y = "Count") +
      theme_minimal() + 
      ylim(histogram_ymin, histogram_ymax)
    print(histogram)
    histogram_plots[[gp_name]] <- histogram
  }
  return(list(normal_plots = normal_plots, histogram_plots = histogram_plots))
}

# Build inducing points prediction matrix
# given inducing points matrix(X) and GP models, predict the labels for the inducing points
build_inducing_pts_pred_matrix <- function(indcpts_matrix, GPmodels, gp_package = 'gplite') {
  # predict_gp <- function(models, data) {
  #   if (gp_package)
  #   predictions <- sapply(models, function(model) gp_pred(model, data, jitter = 1e-4)$mean)
  #   return(predictions)
  # }

  predict_gp <- function(models, data, gp_package = c("gplite", "laGP"), jitter = 1e-4) {
    gp_package <- match.arg(gp_package) # validate gp_package input
    gp_prediction <- switch(gp_package,
      gplite = function(model, data) gp_pred(model, data, jitter = jitter),
      laGP   = function(model, data) predGPsep(model, data)
    )
    predictions <- sapply(models, function(model) gp_prediction(model, data)$mean)
    return(predictions)
  }

  indcpts_pred_matrix <- predict_gp(GPmodels, indcpts_matrix)
  pred_labels <- apply(indcpts_pred_matrix, 1, function(r) which.max(r)-1)

  zero_columns <- matrix(0, nrow = nrow(indcpts_matrix), ncol = 10-length(GPmodels) )

  result_matrix <- cbind(indcpts_matrix, indcpts_pred_matrix, zero_columns, label = pred_labels)
  result_matrix <- as.data.frame(result_matrix)
  result_matrix[] <- lapply(result_matrix, function(x) as.numeric(as.character(x)))
  result_matrix[is.na(result_matrix)] <- 0
  result_matrix <- as.matrix(result_matrix)

  return(result_matrix)
}

###### Save Structured Result: models_store ######
# # models_store is a list with the following structure:
#   models_store <- list(
#     GPs = list(
#       "c0" = list(model = GPmodel_c0, target_class = c("c0", "c5")),
#       "c1" = list(model = GPmodel_c1, target_class = c("c1")),
#       ...),
#     classifiers = list(
#       "c0" = list(classifier = logistic_classifier_c0, target_class = c("c0", "c5"))),
#       ...),
#     metadata = list(
#       existing_class = c("c0", "c1", ... "c5")),
#       existing_GPs = c("c0", "c1", ... "c4"))
#     )
#   )

# Initialize models_store with existing classes and existing GPs, classifiers
construct_models_store <- function(existing_class, existing_GPs, GPmodels, classifiers_list) {
 gp_list <- lapply(existing_GPs, function(gp_name) {
    list(
      model = GPmodels[[gp_name]],
      target_class = c(gp_name)  # default: each GP handles its own class only
    )
  })
  names(gp_list) <- existing_GPs

  if(is.null(classifiers_list)){
    classifiers_list <- list()
  }

  models_store <- list(
    GPs = gp_list,
    classifiers = classifiers_list,
    metadata = list(
      existing_class = existing_class,
      existing_GPs = existing_GPs
    )
  )

  return(models_store)
}

# Save models_store to dictionary as .rda files
save_models_store <- function(models_store, dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path)
    print(paste0("Directory created: ", dir_path))
  }

  # Save each GP model using gp_save
  for (gp_name in names(models_store$GPs)) {
    gp_path <- file.path(dir_path, paste0("GP_", gp_name, ".rda"))
    gp_model <- models_store$GPs[[gp_name]]$model
    gp_save(gp_model, gp_path)
  }

  # TODO: (verify) Save each classifier
  for (clf_name in names(models_store$classifiers)) {
    clf_path <- file.path(dir_path, paste0("classifier_", clf_name, ".rda"))
    clf_obj <- models_store$classifiers[[clf_name]]$classifier
    save(clf_obj, file = clf_path)
  }

  # Save metadata and GP info (excluding actual GP model objects)
  metadata <- models_store$metadata
  GPs_info <- lapply(models_store$GPs, function(gp) {
    list(target_class = gp$target_class)
  })
  classifiers_info <- lapply(models_store$classifiers, function(clf) {
    list(target_class = clf$target_class)
  })
  save(metadata, GPs_info, classifiers_info, file = file.path(dir_path, "models_store_meta.rda"))
}

# Load models_store from directory
load_models_store_rda <- function(dir_path) {
  # Load metadata
  load(file.path(dir_path, "models_store_meta.rda"))  # loads: metadata, GPs_info, classifiers_info

  # Load GPs
  GPs <- list()
  for (gp_name in names(GPs_info)) {
    gp_file <- file.path(dir_path, paste0("GP_", gp_name, ".rda"))
    gp_obj <- gp_load(gp_file)  # loads: gp_obj
    GPs[[gp_name]] <- list(
      model = gp_obj,
      target_class = GPs_info[[gp_name]]$target_class
    )
  }

  # Load classifiers
  classifiers <- list()
  for (clf_name in names(classifiers_info)) {
    load(file.path(dir_path, paste0("classifier_", clf_name, ".rda")))  # loads clf_obj
    classifiers[[clf_name]] <- list(
      model = clf_obj,
      target_class = classifiers_info[[clf_name]]$target_class
    )
  }

  # create models-store structure
  models_store <- list(
    GPs = GPs,
    classifiers = classifiers,
    metadata = metadata
  )

  return(models_store)
}