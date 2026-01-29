## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 8,
  fig.height = 6,
  warning = FALSE,
  message = FALSE
)

## ----install, eval=FALSE------------------------------------------------------
# # From GitHub
# devtools::install_github("evandeilton/OptimalBinningWoE")
# 
# # Install dependencies for this vignette
# install.packages(c("scorecard", "tidymodels", "pROC"))

## ----load_data----------------------------------------------------------------
library(OptimalBinningWoE)
library(scorecard)

# Load German credit dataset
data("germancredit", package = "scorecard")

# Inspect structure
dim(germancredit)
str(germancredit[, 1:8])

# Target variable
table(germancredit$creditability)
cat("\nDefault rate:", round(mean(germancredit$creditability == "bad") * 100, 2), "%\n")

## ----data_prep----------------------------------------------------------------
# Create binary target (must be a factor for tidymodels classification)
german <- germancredit
german$default <- factor(
  ifelse(german$creditability == "bad", 1, 0),
  levels = c(0, 1),
  labels = c("good", "bad")
)
german$creditability <- NULL

# Select key features for demonstration
features_num <- c("duration.in.month", "credit.amount", "age.in.years")
features_cat <- c(
  "status.of.existing.checking.account", "credit.history",
  "purpose", "savings.account.and.bonds"
)

german_model <- german[c("default", features_num, features_cat)]

# Summary statistics
cat("Numerical features:\n")
summary(german_model[, features_num])

cat("\n\nCategorical features:\n")
sapply(german_model[, features_cat], function(x) length(unique(x)))

## ----quickstart_single--------------------------------------------------------
# Bin credit amount with JEDI algorithm
result_single <- obwoe(
  data = german_model,
  target = "default",
  feature = "credit.amount",
  algorithm = "jedi",
  min_bins = 3,
  max_bins = 6
)

# View results
print(result_single)

# Detailed binning table
result_single$results$credit.amount

## ----quickstart_plot----------------------------------------------------------
# WoE pattern visualization
plot(result_single, type = "woe")

## ----quickstart_insights------------------------------------------------------
# Extract metrics
bins <- result_single$results$credit.amount

cat("Binning Summary:\n")
cat("  Number of bins:", nrow(bins), "\n")
cat("  Total IV:", round(sum(bins$iv), 4), "\n")
cat("  Monotonic:", all(diff(bins$woe) >= 0) || all(diff(bins$woe) <= 0), "\n\n")

# Event rates by bin
bins_summary <- data.frame(
  Bin = bins$bin,
  Count = bins$count,
  Event_Rate = round(bins$count_pos / bins$count * 100, 2),
  WoE = round(bins$woe, 4),
  IV_Contribution = round(bins$iv, 4)
)

print(bins_summary)

## ----multifeature_binning-----------------------------------------------------
# Bin all features simultaneously
result_multi <- obwoe(
  data = german_model,
  target = "default",
  algorithm = "cm",
  min_bins = 3,
  max_bins = 4
)

# Summary of all features
summary(result_multi)

## ----feature_selection--------------------------------------------------------
# Extract IV summary
iv_summary <- result_multi$summary[!result_multi$summary$error, ]
iv_summary <- iv_summary[order(-iv_summary$total_iv), ]

# Top predictive features
cat("Top 5 Features by Information Value:\n\n")
print(head(iv_summary[, c("feature", "total_iv", "n_bins")], 5))

# Select features with IV >= 0.02
strong_features <- iv_summary$feature[iv_summary$total_iv >= 0.02]
cat("\n\nFeatures with IV >= 0.02:", length(strong_features), "\n")

## ----gains_analysis-----------------------------------------------------------
# Compute gains for best numerical feature
best_num_feature <- iv_summary$feature[
  iv_summary$feature %in% features_num
][1]

gains <- obwoe_gains(result_multi, feature = best_num_feature, sort_by = "id")

print(gains)

# Plot gains curves
oldpar <- par(mfrow = c(2, 2))
plot(gains, type = "cumulative")
plot(gains, type = "ks")
plot(gains, type = "lift")
plot(gains, type = "woe_iv")
par(oldpar)

## ----algorithm_comparison-----------------------------------------------------
# Test multiple algorithms on credit.amount
algorithms <- c("jedi", "mdlp", "mob", "ewb", "cm")

compare_algos <- function(data, target, feature, algos) {
  results <- lapply(algos, function(algo) {
    tryCatch(
      {
        fit <- obwoe(
          data = data,
          target = target,
          feature = feature,
          algorithm = algo,
          min_bins = 3,
          max_bins = 6
        )

        data.frame(
          Algorithm = algo,
          N_Bins = fit$summary$n_bins[1],
          IV = round(fit$summary$total_iv[1], 4),
          Converged = fit$summary$converged[1],
          stringsAsFactors = FALSE
        )
      },
      error = function(e) {
        # Return NA but log error for debugging during vignette rendering
        message(sprintf("Algorithm '%s' failed: %s", algo, e$message))
        data.frame(
          Algorithm = algo,
          N_Bins = NA_integer_,
          IV = NA_real_,
          Converged = FALSE,
          stringsAsFactors = FALSE
        )
      }
    )
  })

  do.call(rbind, results)
}

# Compare on credit.amount
comp_result <- compare_algos(
  german_model,
  "default",
  "credit.amount",
  algorithms
)

cat("Algorithm Comparison on 'credit.amount':\n\n")
print(comp_result[order(-comp_result$IV), ])

## ----algo_guide---------------------------------------------------------------
# View algorithm capabilities
algo_info <- obwoe_algorithms()

cat("Algorithm Categories:\n\n")

cat("Fast for Large Data (O(n) complexity):\n")
print(algo_info[
  algo_info$algorithm %in% c("ewb", "sketch"),
  c("algorithm", "numerical", "categorical")
])

cat("\n\nRegulatory Compliant (Monotonic):\n")
print(algo_info[
  algo_info$algorithm %in% c("mob", "mblp", "ir"),
  c("algorithm", "numerical", "categorical")
])

cat("\n\nGeneral Purpose (algorithm):\n")
print(algo_info[
  algo_info$name %in% c("jedi", "cm", "mdlp"),
  c("algorithm", "numerical", "categorical")
])

## ----tidymodels_setup, message=FALSE------------------------------------------
library(tidymodels)

# Train/test split with stratification
set.seed(123)
german_split <- initial_split(german_model, prop = 0.7, strata = default)
train_data <- training(german_split)
test_data <- testing(german_split)

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")
cat("Train default rate:", round(mean(train_data$default == "bad") * 100, 2), "%\n")

## ----recipe_definition--------------------------------------------------------
# Create recipe with WoE transformation
rec_woe <- recipe(default ~ ., data = train_data) %>%
  step_obwoe(
    all_predictors(),
    outcome = "default",
    algorithm = "jedi",
    min_bins = 2,
    max_bins = tune(), # Hyperparameter tuning
    bin_cutoff = 0.05,
    output = "woe"
  )

# Preview recipe
rec_woe

## ----workflow_setup-----------------------------------------------------------
# Logistic regression specification
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Create complete workflow
wf_credit <- workflow() %>%
  add_recipe(rec_woe) %>%
  add_model(lr_spec)

wf_credit

## ----cv_tuning----------------------------------------------------------------
# Define tuning grid
tune_grid <- tibble(max_bins = c(4, 6, 8))

# Create cross-validation folds
set.seed(456)
cv_folds <- vfold_cv(train_data, v = 5, strata = default)

# Tune workflow
tune_results <- tune_grid(
  wf_credit,
  resamples = cv_folds,
  grid = tune_grid,
  metrics = metric_set(roc_auc, accuracy)
)

# Best configuration
collect_metrics(tune_results) %>%
  # filter(.metric == "roc_auc") %>%
  arrange(desc(mean))

# Visualize tuning
autoplot(tune_results, metric = "roc_auc")

## ----final_model--------------------------------------------------------------
# Select best parameters
best_params <- select_best(tune_results, metric = "roc_auc")
cat("Optimal max_bins:", best_params$max_bins, "\n\n")

# Finalize and fit
final_wf <- finalize_workflow(wf_credit, best_params)
final_fit <- fit(final_wf, data = train_data)

# Extract coefficients
final_fit %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(desc(abs(estimate)))

## ----model_eval---------------------------------------------------------------
# Predictions on test set
test_pred <- augment(final_fit, test_data)

# Performance metrics
metrics <- metric_set(roc_auc, accuracy, sens, spec, precision)
metrics(test_pred,
  truth = default, estimate = .pred_class,
  .pred_bad, event_level = "second"
)

# ROC curve
roc_curve(test_pred,
  truth = default, .pred_bad,
  event_level = "second"
) %>%
  autoplot() +
  labs(title = "ROC Curve - German Credit Model")

## ----inspect_binning----------------------------------------------------------
# Extract trained recipe
trained_rec <- extract_recipe(final_fit)
woe_step <- trained_rec$steps[[1]]

# View binning for credit.amount
credit_bins <- woe_step$binning_results$credit.amount

data.frame(
  Bin = credit_bins$bin,
  WoE = round(credit_bins$woe, 4),
  IV = round(credit_bins$iv, 4)
)

## ----scorecard_split----------------------------------------------------------
set.seed(789)
n_total <- nrow(german_model)
train_idx <- sample(1:n_total, size = 0.7 * n_total)

train_sc <- german_model[train_idx, ]
test_sc <- german_model[-train_idx, ]

## ----scorecard_binning--------------------------------------------------------
# Use monotonic binning for regulatory compliance
sc_binning <- obwoe(
  data = train_sc,
  target = "default",
  algorithm = "mob", # Monotonic Optimal Binning
  min_bins = 3,
  max_bins = 5,
  control = control.obwoe(
    bin_cutoff = 0.05,
    convergence_threshold = 1e-6
  )
)

summary(sc_binning)

## ----scorecard_transform------------------------------------------------------
# Transform training data with error handling
train_woe <- tryCatch(
  {
    obwoe_apply(train_sc, sc_binning, keep_original = FALSE)
  },
  error = function(e) {
    message("Error in obwoe_apply for training data: ", e$message)
    message("This may occur with certain data distributions. Skipping transformation.")
    return(NULL)
  }
)

# Only proceed if transformation succeeded
if (!is.null(train_woe)) {
  # Transform test data (uses training bins)
  test_woe <- obwoe_apply(test_sc, sc_binning, keep_original = FALSE)

  # Preview transformed features
  head(train_woe[, c("default", grep("_woe$", names(train_woe), value = TRUE)[1:3])], 10)
} else {
  message("Skipping WoE transformation demonstration due to data incompatibility.")
}

## ----scorecard_model----------------------------------------------------------
if (!is.null(train_woe)) {
  # Select features with IV >= 0.02
  selected <- sc_binning$summary$feature[
    sc_binning$summary$total_iv >= 0.02 &
      !sc_binning$summary$error
  ]

  woe_vars <- paste0(selected, "_woe")
  formula_str <- paste("default ~", paste(woe_vars, collapse = " + "))

  # Fit model
  scorecard_glm <- glm(
    as.formula(formula_str),
    data = train_woe,
    family = binomial(link = "logit")
  )

  summary(scorecard_glm)
} else {
  message("Skipping model building - WoE transformation failed.")
}

## ----scorecard_validation-----------------------------------------------------
if (!is.null(train_woe) && exists("scorecard_glm")) {
  library(pROC)

  # Predictions
  test_woe$score <- predict(scorecard_glm, newdata = test_woe, type = "response")

  # ROC curve
  roc_obj <- roc(test_woe$default, test_woe$score, quiet = TRUE)
  auc_val <- auc(roc_obj)

  # KS statistic
  ks_stat <- max(abs(
    ecdf(test_woe$score[test_woe$default == "bad"])(seq(0, 1, 0.01)) -
      ecdf(test_woe$score[test_woe$default == "good"])(seq(0, 1, 0.01))
  ))

  # Gini coefficient
  gini <- 2 * auc_val - 1

  cat("Scorecard Performance:\n")
  cat("  AUC:  ", round(auc_val, 4), "\n")
  cat("  Gini: ", round(gini, 4), "\n")
  cat("  KS:   ", round(ks_stat * 100, 2), "%\n")

  # ROC plot
  plot(roc_obj,
    main = "Scorecard ROC Curve",
    print.auc = TRUE, print.thres = "best"
  )
} else {
  message("Skipping validation - model not available.")
}

## ----preprocessing------------------------------------------------------------
# Simulate feature with issues
set.seed(2024)
problematic <- c(
  rnorm(800, 5000, 2000), # Normal values
  rep(NA, 100), # Missing
  runif(100, -10000, 50000) # Outliers
)

target_sim <- rbinom(1000, 1, 0.3)

# Preprocess with IQR method
preproc_result <- ob_preprocess(
  feature = problematic,
  target = target_sim,
  outlier_method = "iqr",
  outlier_process = TRUE,
  preprocess = "both"
)

# View report
print(preproc_result$report)

# Compare distributions
cat("\n\nBefore preprocessing:\n")
cat("  Range:", range(problematic, na.rm = TRUE), "\n")
cat("  Missing:", sum(is.na(problematic)), "\n")
cat("  Mean:", round(mean(problematic, na.rm = TRUE), 2), "\n")

cat("\nAfter preprocessing:\n")
cleaned <- preproc_result$preprocess$feature_preprocessed
cat("  Range:", range(cleaned), "\n")
cat("  Missing:", sum(is.na(cleaned)), "\n")
cat("  Mean:", round(mean(cleaned), 2), "\n")

## ----production_save, eval=FALSE----------------------------------------------
# # Add metadata to model
# sc_binning$metadata <- list(
#   creation_date = Sys.time(),
#   creator = Sys.info()["user"],
#   dataset_size = nrow(train_sc),
#   default_rate = mean(train_sc$default == "bad"),
#   r_version = R.version.string,
#   package_version = packageVersion("OptimalBinningWoE")
# )
# 
# # Save model
# saveRDS(sc_binning, "credit_scorecard_v1_20250101.rds")
# 
# # Load model
# loaded_model <- readRDS("credit_scorecard_v1_20250101.rds")

## ----production_score, eval=FALSE---------------------------------------------
# score_applications <- function(new_data, model_file) {
#   # Load binning model
#   binning_model <- readRDS(model_file)
# 
#   # Validate required features
#   required_vars <- binning_model$summary$feature[
#     !binning_model$summary$error
#   ]
# 
#   missing_vars <- setdiff(required_vars, names(new_data))
#   if (length(missing_vars) > 0) {
#     stop("Missing features: ", paste(missing_vars, collapse = ", "))
#   }
# 
#   # Apply WoE transformation
#   scored <- obwoe_apply(new_data, binning_model, keep_original = TRUE)
# 
#   # Add timestamp
#   scored$scoring_date <- Sys.Date()
# 
#   return(scored)
# }
# 
# # Usage example
# # new_apps <- read.csv("new_applications.csv")
# # scored_apps <- score_applications(new_apps, "credit_scorecard_v1_20250101.rds")

## ----pitfalls, eval=FALSE-----------------------------------------------------
# # ❌ Don't bin on full dataset before splitting
# # This causes data leakage!
# bad_approach <- obwoe(full_data, target = "y")
# train_woe <- obwoe_apply(train_data, bad_approach)
# 
# # ✅ Correct: Bin only on training data
# good_approach <- obwoe(train_data, target = "y")
# test_woe <- obwoe_apply(test_data, good_approach)
# 
# # ❌ Don't ignore IV thresholds
# # IV > 0.50 likely indicates target leakage
# suspicious_features <- result$summary$feature[
#   result$summary$total_iv > 0.50
# ]
# 
# # ❌ Don't over-bin
# # Too many bins (>10) reduces interpretability
# # and may cause overfitting

## ----session_info-------------------------------------------------------------
sessionInfo()

