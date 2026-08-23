## ----setup, include = FALSE-----------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7.5,
  fig.height = 4.5,
  fig.align = "center",
  warning = FALSE,
  message = FALSE
)
options(width = 100, digits = 4)

## ----libs-----------------------------------------------------------------------------------------
library(OptimalBinningWoE)
library(recipes)
set.seed(20260819)

## ----data-generator-------------------------------------------------------------------------------
make_base <- function(n, vintage) {
  age    <- pmax(18, round(rnorm(n, 41, 13)))
  income <- round(exp(rnorm(n, 8.1, 0.55)))
  tenure <- pmax(0, round(rexp(n, 1 / 48)))
  util   <- pmin(1.6, pmax(0, rbeta(n, 2, 4) + rnorm(n, 0, 0.08)))
  inq    <- rpois(n, 1.3)
  dlq    <- rpois(n, 0.35)
  bureau <- round(rnorm(n, 640, 85))
  ltv    <- pmin(1.3, pmax(0.1, rnorm(n, 0.72, 0.16)))

  region  <- sample(c("N", "NE", "CO", "SE", "S"), n, TRUE, c(.09, .27, .07, .42, .15))
  channel <- sample(c("branch", "broker", "digital", "partner"), n, TRUE, c(.34, .21, .33, .12))
  product <- sample(c("auto", "personal", "payroll", "card"), n, TRUE, c(.28, .35, .22, .15))
  occ     <- sample(c("salaried", "self_employed", "retired", "public", "informal"), n, TRUE)
  housing <- sample(c("owned", "rented", "family", "mortgaged"), n, TRUE, c(.31, .34, .20, .15))
  dealer  <- sample(c(LETTERS[1:3], paste0("Z", 1:14)), n, TRUE,
                    c(rep(.30, 3), rep(.10 / 14, 14)))

  lp <- -3.80 -
    0.019 * (age - 41) -
    0.55 * scale(log(income))[, 1] +
    1.35 * util +
    0.24 * inq + 0.42 * dlq -
    0.008 * (bureau - 640) +
    1.70 * ltv +
    0.30 * (channel == "broker") - 0.22 * (channel == "branch") +
    0.55 * (occ == "informal") - 0.40 * (occ == "public") +
    0.45 * (housing == "rented") - 0.006 * pmin(tenure, 120)
  y <- rbinom(n, 1, plogis(lp))

  # a second bureau vendor and a declared-income field: near-duplicates of
  # variables already in the base, which is what feature stores actually hand you
  bureau_alt <- round(0.80 * bureau + 0.20 * rnorm(n, 640, 85) + rnorm(n, 0, 35))
  income_declared <- round(income * exp(rnorm(n, 0, 0.15)))

  df <- data.frame(
    age, income, tenure_months = tenure, utilisation = util,
    inquiries_6m = inq, delinq_12m = dlq, bureau_score = bureau, ltv,
    bureau_alt, income_declared,
    region, channel, product, occupation = occ, housing,
    dealer_code = dealer, stringsAsFactors = FALSE
  )

  # eight columns of pure noise and four uninformative flags
  for (j in 1:8) df[[sprintf("noise_%02d", j)]] <- rnorm(n)
  for (j in 1:4) df[[sprintf("flag_%02d", j)]] <- sample(c("Y", "N"), n, TRUE)

  # populated only after booking: unavailable at decision time
  df$collections_after_booking <- ifelse(y == 1, rpois(n, 2.2), rpois(n, 0.05))

  df$utilisation[sample(n, n * 0.12)] <- NA
  df$tenure_months[sample(n, n * 0.07)] <- NA
  df$occupation[sample(n, n * 0.05)] <- NA

  df$vintage <- vintage
  df$default <- y
  df
}

## ----data-build-----------------------------------------------------------------------------------
dev <- make_base(20000, "2024H1")
oot <- make_base(8000, "2024H2")

predictors <- setdiff(names(dev), c("default", "vintage"))
c(dev = nrow(dev), oot = nrow(oot), predictors = length(predictors))
c(dev_rate = mean(dev$default), oot_rate = mean(oot$default))

## ----missing--------------------------------------------------------------------------------------
as_levels <- function(df) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], function(v) replace(v, is.na(v), -999))
  df[!num] <- lapply(df[!num], function(v) replace(v, is.na(v), "MISSING"))
  df
}

dev <- as_levels(dev)
oot <- as_levels(oot)

## ----fit------------------------------------------------------------------------------------------
binning <- obwoe(dev, target = "default", feature = predictors,
                 min_bins = 2, max_bins = 6)
binning

## ----select---------------------------------------------------------------------------------------
sel <- obwoe_select(
  binning,
  iv_min            = 0.02,
  iv_max            = 0.50,
  require_monotonic = "numeric",
  min_bin_pct       = 0.03,
  sort_by           = "iv"
)

head(sel[, c("feature", "type", "n_bins", "total_iv", "iv_class",
             "ks", "monotonic", "quality", "selected")], 12)

## ----select-reasons-------------------------------------------------------------------------------
table(sel$reason)

## ----select-leak----------------------------------------------------------------------------------
sel[sel$reason != "OK" & sel$total_iv > 0.05,
    c("feature", "total_iv", "iv_class", "n_degenerate_bins", "reason")]

## ----select-rare----------------------------------------------------------------------------------
sel[grepl("SMALL_BIN", sel$reason),
    c("feature", "n_bins", "min_bin_pct", "min_bin_count", "reason")]

## ----shortlist------------------------------------------------------------------------------------
shortlist <- sel$feature[sel$selected]
shortlist

## ----evidence-------------------------------------------------------------------------------------
evidence <- obwoe_select(binning, detail = "full")
evidence[evidence$feature == "bureau_score",
         c("bin", "count", "pos", "pos_rate", "woe", "iv", "lift")]

## ----corr-----------------------------------------------------------------------------------------
woe_dev <- obwoe_apply(dev, binning, keep_original = FALSE)
pairs <- obcorr(woe_dev[, paste0(shortlist, "_woe")], method = "pearson")

head(pairs[order(-abs(pairs$pearson)), ], 5)

## ----prune----------------------------------------------------------------------------------------
prune <- function(pairs, ranking, cutoff = 0.70) {
  hits <- pairs[abs(pairs$pearson) >= cutoff, , drop = FALSE]
  weaker <- mapply(function(a, b) {
    c(a, b)[which.max(c(match(a, ranking), match(b, ranking)))]
  }, hits$x, hits$y)
  unique(as.character(weaker))
}

ranking <- paste0(shortlist, "_woe")
dropped <- prune(pairs, ranking, cutoff = 0.70)

final_vars <- setdiff(shortlist, sub("_woe$", "", dropped))
dropped
c(shortlist = length(shortlist), dropped = length(dropped),
  final = length(final_vars))

## ----recipe---------------------------------------------------------------------------------------
dev$default_f <- factor(dev$default, levels = c(0, 1))
oot$default_f <- factor(oot$default, levels = c(0, 1))

form <- reformulate(final_vars, response = "default_f")

rec <- recipe(form, data = dev) |>
  step_obwoe(all_predictors(), outcome = "default_f",
             min_bins = 2, max_bins = 6, bin_cutoff = 0.03,
             output = "woe")

prepped <- prep(rec, training = dev)
prepped

## ----recipe-tidy----------------------------------------------------------------------------------
rules <- tidy(prepped, number = 1)
head(rules, 8)
nrow(rules)

## ----bake-----------------------------------------------------------------------------------------
train_woe <- bake(prepped, new_data = dev)
oot_woe   <- bake(prepped, new_data = oot)

head(train_woe, 3)

## ----model----------------------------------------------------------------------------------------
fit <- glm(default_f ~ ., data = train_woe, family = binomial())
round(coef(summary(fit)), 4)

## ----model-signs----------------------------------------------------------------------------------
sum(coef(fit)[-1] < 0)

## ----points---------------------------------------------------------------------------------------
pdo <- 20
factor_ <- pdo / log(2)
offset_ <- 600 - factor_ * log(50)

to_score <- function(link) round(offset_ - factor_ * link)

dev$score <- to_score(predict(fit, newdata = train_woe, type = "link"))
oot$score <- to_score(predict(fit, newdata = oot_woe, type = "link"))

summary(dev$score)

## ----points-decomposition-------------------------------------------------------------------------
lead_var <- final_vars[1]
per_bin <- rules[rules$terms == lead_var, c("bin", "woe")]
per_bin$points <- round(-factor_ * coef(fit)[[lead_var]] * per_bin$woe)

lead_var
per_bin

## ----gains----------------------------------------------------------------------------------------
gains_oot <- obwoe_gains(oot, target = "default", feature = "score",
                         use_column = "direct", n_groups = 10, sort_by = "bin")
gains_oot

## ----gains-dev------------------------------------------------------------------------------------
gains_dev <- obwoe_gains(dev, target = "default", feature = "score",
                         use_column = "direct", n_groups = 10, sort_by = "bin")

data.frame(
  sample = c("development", "out-of-time"),
  ks     = round(c(gains_dev$metrics$ks, gains_oot$metrics$ks), 2),
  gini   = round(c(gains_dev$metrics$gini, gains_oot$metrics$gini), 2),
  auc    = round(c(gains_dev$metrics$auc, gains_oot$metrics$auc), 4)
)

## ----gains-plot, fig.height=6---------------------------------------------------------------------
op <- par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(gains_oot, type = "cumulative")
plot(gains_oot, type = "ks")
plot(gains_oot, type = "lift")
plot(gains_oot, type = "woe_iv")
par(op)

## ----psi------------------------------------------------------------------------------------------
psi <- function(p, q) {
  p <- pmax(p, 1e-6)
  q <- pmax(q, 1e-6)
  sum((p - q) * log(p / q))
}

share <- function(x, levels) as.numeric(table(factor(x, levels))) / length(x)

bins_dev <- obwoe_apply(dev, binning, keep_original = FALSE)
bins_oot <- obwoe_apply(oot, binning, keep_original = FALSE)

psi_vars <- vapply(final_vars, function(v) {
  levels <- binning$results[[v]]$bin
  psi(share(bins_dev[[paste0(v, "_bin")]], levels),
      share(bins_oot[[paste0(v, "_bin")]], levels))
}, numeric(1))

score_cuts <- c(-Inf, quantile(dev$score, seq(0.1, 0.9, 0.1)), Inf)
psi_score <- psi(share(cut(dev$score, score_cuts), levels(cut(dev$score, score_cuts))),
                 share(cut(oot$score, score_cuts), levels(cut(dev$score, score_cuts))))

psi_table <- data.frame(
  variable = c("SCORE", final_vars),
  psi = round(c(psi_score, psi_vars), 4),
  row.names = NULL
)
psi_table <- psi_table[order(-psi_table$psi), ]
row.names(psi_table) <- NULL
psi_table

## ----sql------------------------------------------------------------------------------------------
sql <- obwoe_sql(
  binning,
  table        = "risk.applications",
  features     = final_vars,
  keep_columns = c("application_id", "vintage"),
  dialect      = "postgres",
  style        = "view",
  view_name    = "risk.v_application_woe"
)

writeLines(head(strsplit(as.character(sql), "\n")[[1]], 28))

## ----sql-file, eval=FALSE-------------------------------------------------------------------------
# obwoe_sql(binning, table = "risk.applications", features = final_vars,
#           dialect = "postgres", file = "woe_transform.sql")

## ----sql-coefs------------------------------------------------------------------------------------
data.frame(
  variable = names(coef(fit)),
  beta     = round(as.numeric(coef(fit)), 6),
  row.names = NULL
)

## ----deploy-r, eval=FALSE-------------------------------------------------------------------------
# artefact <- list(
#   recipe       = prepped,
#   coefficients = coef(fit),
#   scaling      = c(factor = factor_, offset = offset_),
#   screening    = sel,
#   built_on     = Sys.Date(),
#   package      = as.character(utils::packageVersion("OptimalBinningWoE"))
# )
# saveRDS(artefact, "scorecard_v1.rds")
# 
# score_batch <- function(new_data, artefact) {
#   woe <- bake(artefact$recipe, new_data = new_data)
#   lp <- as.numeric(cbind(1, as.matrix(woe[names(artefact$coefficients)[-1]])) %*%
#                      artefact$coefficients)
#   round(artefact$scaling[["offset"]] - artefact$scaling[["factor"]] * lp)
# }

## ----tune, eval=FALSE-----------------------------------------------------------------------------
# library(tidymodels)
# 
# rec_tune <- recipe(form, data = dev) |>
#   step_obwoe(all_predictors(), outcome = "default_f",
#              max_bins = tune(), bin_cutoff = tune())
# 
# wf <- workflow() |>
#   add_recipe(rec_tune) |>
#   add_model(logistic_reg() |> set_engine("glm"))
# 
# grid <- grid_regular(obwoe_max_bins(range = c(3L, 10L)),
#                      obwoe_bin_cutoff(range = c(0.01, 0.10)),
#                      levels = 4)
# 
# folds <- vfold_cv(dev, v = 5, strata = default_f)
# 
# tuned <- tune_grid(wf, resamples = folds, grid = grid,
#                    metrics = metric_set(roc_auc))
# 
# final_wf <- finalize_workflow(wf, select_best(tuned, metric = "roc_auc"))
# final_fit <- fit(final_wf, data = dev)

## ----session--------------------------------------------------------------------------------------
sessionInfo()

