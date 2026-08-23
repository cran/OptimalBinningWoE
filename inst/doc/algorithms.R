## ----setup, include = FALSE-----------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE
)
options(width = 100, digits = 4)

## ----libs-----------------------------------------------------------------------------------------
library(OptimalBinningWoE)

## ----registry-------------------------------------------------------------------------------------
alg <- obwoe_algorithms()
c(algorithms = nrow(alg),
  numerical = sum(alg$numerical),
  categorical = sum(alg$categorical),
  both = sum(alg$numerical & alg$categorical),
  multinomial = sum(alg$multinomial))

## ----via-obwoe------------------------------------------------------------------------------------
data_path <- system.file("extdata", "germancredit.csv.gz",
                         package = "OptimalBinningWoE")
gc_data <- utils::read.csv(gzfile(data_path), stringsAsFactors = FALSE)
gc_data$target <- as.integer(1L - gc_data$credit_risk)
gc_data$credit_risk <- NULL

fit <- obwoe(gc_data, target = "target", feature = c("duration", "purpose"),
             algorithm = "jedi", min_bins = 2, max_bins = 5)
fit$summary[, c("feature", "type", "algorithm", "n_bins", "total_iv")]

## ----via-wrapper----------------------------------------------------------------------------------
direct <- ob_numerical_jedi(target = gc_data$target,
                            feature = gc_data$duration,
                            min_bins = 2, max_bins = 5)
data.frame(bin = direct$bin, count = direct$count,
           woe = round(direct$woe, 4), iv = round(direct$iv, 4))

## ----forwarding-----------------------------------------------------------------------------------
via_obwoe <- obwoe(gc_data, target = "target", feature = "duration",
                   algorithm = "dmiv",
                   control = control.obwoe(divergence_method = "kl"))
direct_dmiv <- ob_numerical_dmiv(target = gc_data$target,
                                 feature = gc_data$duration,
                                 divergence_method = "kl")

c(`through obwoe()` = via_obwoe$results$duration$divergence_method,
  `direct wrapper`  = direct_dmiv$divergence_method)

## ----coverage-------------------------------------------------------------------------------------
coverage <- alg
coverage$feature_types <- ifelse(coverage$numerical & coverage$categorical, "both",
                          ifelse(coverage$numerical, "numerical", "categorical"))
coverage$target <- ifelse(coverage$multinomial, "binary + multinomial", "binary")
coverage[order(coverage$feature_types, coverage$algorithm),
         c("algorithm", "feature_types", "target")]

## ----bench, results = "asis"----------------------------------------------------------------------
run_all <- function(feature, type) {
  ids <- alg$algorithm[alg[[type]]]
  out <- lapply(ids, function(a) {
    r <- try(suppressWarnings(suppressMessages(
      obwoe(gc_data, target = "target", feature = feature,
            algorithm = a, min_bins = 2, max_bins = 5))), silent = TRUE)
    if (inherits(r, "try-error") || r$summary$error) {
      data.frame(algorithm = a, bins = NA_integer_, total_iv = NA_real_)
    } else {
      data.frame(algorithm = a, bins = r$summary$n_bins,
                 total_iv = round(r$summary$total_iv, 4))
    }
  })
  do.call(rbind, out)
}
num_res <- run_all("duration", "numerical")
knitr::kable(num_res[order(-num_res$total_iv), ], row.names = FALSE,
             caption = "Numerical engines on `duration`")

## ----bench-cat, results = "asis"------------------------------------------------------------------
cat_res <- run_all("purpose", "categorical")
knitr::kable(cat_res[order(-cat_res$total_iv), ], row.names = FALSE,
             caption = "Categorical engines on `purpose`")

