## ----setup, include = FALSE-----------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7.5,
  fig.height = 5,
  fig.align = "center",
  warning = FALSE,
  message = FALSE
)
options(width = 100, digits = 4)

## ----library--------------------------------------------------------------------------------------
library(OptimalBinningWoE)

## ----data-----------------------------------------------------------------------------------------
german <- read.csv(
  gzfile(system.file("extdata", "germancredit.csv.gz",
                     package = "OptimalBinningWoE")),
  stringsAsFactors = FALSE
)

# credit_risk is 1 for a good customer; the event we model is default
german$default <- 1L - german$credit_risk
german$credit_risk <- NULL

dim(german)
table(german$default)

## ----data-glimpse---------------------------------------------------------------------------------
str(german[, c("duration", "amount", "age", "purpose", "savings")])

## ----single-fit-----------------------------------------------------------------------------------
fit <- obwoe(german, target = "default", feature = "duration",
             min_bins = 3, max_bins = 6)
fit

## ----single-result--------------------------------------------------------------------------------
res <- fit$results$duration
data.frame(
  bin   = res$bin,
  count = res$count,
  pos   = res$count_pos,
  rate  = round(res$count_pos / res$count, 4),
  woe   = round(res$woe, 4),
  iv    = round(res$iv, 4)
)

## ----single-cutpoints-----------------------------------------------------------------------------
res$cutpoints

## ----single-plot, fig.height=4.5------------------------------------------------------------------
plot(fit, type = "woe", feature = "duration")

## ----gains----------------------------------------------------------------------------------------
gains <- obwoe_gains(fit, feature = "duration", sort_by = "woe")
gains

## ----gains-plot, fig.height=6---------------------------------------------------------------------
op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(gains, type = "cumulative")
plot(gains, type = "ks")
plot(gains, type = "lift")
plot(gains, type = "woe_iv")
par(op)

## ----multi-fit------------------------------------------------------------------------------------
model <- obwoe(german, target = "default", min_bins = 2, max_bins = 6)
model

## ----multi-summary--------------------------------------------------------------------------------
summary(model)

## ----select---------------------------------------------------------------------------------------
sel <- obwoe_select(model)
head(sel[, c("feature", "type", "n_bins", "total_iv", "iv_class",
             "ks", "gini", "monotonic", "quality", "selected")], 10)

## ----select-rejected------------------------------------------------------------------------------
sel[!sel$selected, c("feature", "total_iv", "iv_class", "reason")]

## ----select-admit---------------------------------------------------------------------------------
admitted <- obwoe_select(model, iv_max = Inf)
admitted[admitted$feature == "status",
         c("feature", "total_iv", "quality", "selected", "reason")]

## ----select-policy--------------------------------------------------------------------------------
strict <- obwoe_select(
  model,
  iv_min            = 0.02,      # drop the Unpredictive band
  iv_max            = 0.50,      # drop the Suspicious band
  require_monotonic = "numeric", # ordering is intrinsic only for numerics
  monotonicity      = "strict",  # no ties between adjacent bins
  min_bin_pct       = 0.05,      # every bin holds at least 5% of the base
  allow_degenerate  = FALSE,     # no bin without events or without non-events
  top_n             = 8,
  sort_by           = "ks"
)
table(strict$reason)

## ----select-full----------------------------------------------------------------------------------
detail <- obwoe_select(model, detail = "full")
dim(detail)

detail[detail$feature == "savings",
       c("bin", "n_categories", "count", "pos_rate", "woe", "iv", "lift")]

## ----algo-list------------------------------------------------------------------------------------
algos <- obwoe_algorithms()
table(numerical = algos$numerical, categorical = algos$categorical)

## ----algo-compare---------------------------------------------------------------------------------
compare <- function(alg) {
  f <- obwoe(german, target = "default", feature = "amount",
             algorithm = alg, min_bins = 2, max_bins = 6)
  s <- obwoe_select(f, require_monotonic = "none")
  data.frame(algorithm = alg, n_bins = s$n_bins, iv = round(s$total_iv, 4),
             ks = round(s$ks, 4), monotonic = s$monotonic)
}

do.call(rbind, lapply(c("jedi", "mdlp", "mob", "ir", "dp"), compare))

## ----apply----------------------------------------------------------------------------------------
scored <- obwoe_apply(german, model, keep_original = FALSE)
head(scored[, c("default", "duration_bin", "duration_woe",
                "purpose_bin", "purpose_woe")], 4)

## ----apply-newdata--------------------------------------------------------------------------------
two <- obwoe(german, target = "default",
             feature = c("duration", "purpose"), max_bins = 6)

new_data <- data.frame(
  duration = c(4, 10, 200, NA),
  purpose  = c("car (new)", "unseen category", "education", NA)
)
obwoe_apply(new_data, two, keep_original = TRUE)

## ----apply-model----------------------------------------------------------------------------------
keep <- sel$feature[sel$selected]
woe_cols <- paste0(keep, "_woe")
train <- scored[, c("default", woe_cols)]

glm_fit <- glm(default ~ ., data = train, family = binomial())
round(head(coef(summary(glm_fit)), 6), 4)

## ----sql------------------------------------------------------------------------------------------
obwoe_sql(
  model,
  table        = "risk.applications",
  features     = c("duration", "purpose"),
  keep_columns = "application_id",
  dialect      = "postgres"
)

## ----sql-case-------------------------------------------------------------------------------------
obwoe_sql(model, features = "age", style = "case", comment = FALSE)

## ----preprocess-----------------------------------------------------------------------------------
set.seed(2024)
messy <- c(rnorm(800, 5000, 2000), rep(NA, 100), runif(100, -1e4, 5e4))
y <- rbinom(1000, 1, 0.3)

prep <- ob_preprocess(
  feature         = messy,
  target          = y,
  outlier_method  = "iqr",
  outlier_process = TRUE,
  preprocess      = "both"
)

prep$report

## ----session--------------------------------------------------------------------------------------
sessionInfo()

