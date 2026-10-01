# C-index boosting with stability selection (Mayr, Hofner & Schmid 2016,
# BMC Bioinformatics 17:288) as an additional survival baseline (reviewer point 3).
# Uses the SAME saved outer splits and writes records in the same format as
# run_selection.py, so run_evaluation.py, the report scripts and stability.py pick
# them up unchanged. Preprocessing (median imputation, scaling) uses the training
# part of each split only.
#
# Usage:
#   Rscript r/cindex_boost_stabsel.R <dataset> [split_id ...]
# Environment (optional):
#   STABSEL_CORES  cores for the stability-selection refits (default 1)
#   CINDEX_MSTOP   boosting iterations of the base model   (default 1500)
#
# Cost: the smoothed C-index is O(n^2) per boosting iteration and stability
# selection refits the model 100 times on half-samples, so run time grows with the
# square of the training size. See README for which datasets are feasible.
#
# mstop must be large enough for every half-sample fit to reach q selected
# variables (with nu = 0.1 the 4th variable entered at iteration ~800 on WHAS).
# If stabsel warns that mstop was too small, the split is retried with 2x and 4x
# mstop; if it still fails, the split is recorded as failed (never silently wrong).
suppressPackageStartupMessages({
  library(jsonlite); library(survival); library(mboost); library(stabs)
})
args    <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript r/cindex_boost_stabsel.R <dataset> [split_id ...]")
dataset <- args[1]
only    <- args[-1]
cores   <- as.integer(Sys.getenv("STABSEL_CORES", "1"))
mstop   <- as.integer(Sys.getenv("CINDEX_MSTOP", "1500"))
options(mc.cores = cores)

split_file <- sprintf("splits/%s.json", dataset)
if (!file.exists(split_file))
  stop(split_file, " not found: run run_selection.py (or _concurrent) for this dataset first")

df     <- read.csv(sprintf("datasets/%s_cleaned.csv", dataset), check.names = FALSE)
time   <- df$lower_bound
event  <- as.integer(is.finite(df$upper_bound))
X      <- as.matrix(df[, setdiff(names(df), c("lower_bound", "upper_bound"))])
safe   <- make.names(colnames(X), unique = TRUE)  # identical to data.frame(check.names=TRUE)
splits <- fromJSON(split_file, simplifyVector = FALSE)
method <- "CIndexBoost-StabSel"

for (s in splits) {
  if (length(only) && !(s$split_id %in% only)) next
  out <- sprintf("results/selection/%s/%s/%s.json", dataset, method, s$split_id)
  if (file.exists(out)) next
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  tr  <- unlist(s$train) + 1                       # python -> R indexing
  Xtr <- X[tr, , drop = FALSE]
  med <- apply(Xtr, 2, median, na.rm = TRUE)       # training-fold preprocessing only
  for (j in seq_len(ncol(Xtr))) Xtr[is.na(Xtr[, j]), j] <- med[j]
  sds <- apply(Xtr, 2, sd); sds[!is.finite(sds) | sds == 0] <- 1
  Xtr <- scale(Xtr, center = colMeans(Xtr), scale = sds)
  colnames(Xtr) <- safe
  d <- data.frame(Xtr, check.names = FALSE)
  d$time <- time[tr]; d$event <- event[tr]

  set.seed(s$seed)
  t0 <- proc.time()
  rec <- tryCatch({
    q  <- min(ceiling(sqrt(0.8 * ncol(Xtr))), ncol(Xtr) - 1)
    st <- NULL
    for (m in mstop * c(1, 2, 4)) {
      fit <- glmboost(Surv(time, event) ~ ., data = d, family = Cindex(sigma = 0.1),
                      control = boost_control(mstop = m, nu = 0.1))
      too_small <- FALSE
      st <- withCallingHandlers(
        stabsel(fit, q = q, PFER = 1, sampling.type = "SS"),
        warning = function(w) {
          if (grepl("too small", conditionMessage(w))) too_small <<- TRUE
          invokeRestart("muffleWarning")
        })
      if (!too_small) break
      st <- NULL
    }
    if (is.null(st)) stop(sprintf("mstop %d too small to reach q = %d variables", m, q))
    sel <- names(sort(st$max[st$selected], decreasing = TRUE))
    idx <- match(sel, safe) - 1                     # back to 0-based column ids
    if (anyNA(idx)) stop("could not map selected variable names back to columns")
    list(status = "ok", features = as.list(idx), n_selected = length(idx),
         q = q, cutoff = st$cutoff, mstop_used = m)
  }, error = function(e) list(status = "failed", error = conditionMessage(e),
                              features = list(), n_selected = 0))
  el <- proc.time() - t0
  rec <- c(list(split_id = s$split_id, "repeat" = s$`repeat`, fold = s$fold, seed = s$seed,
                test_hash = s$test_hash, dataset = dataset, task = "surv", method = method,
                code_version = "R", wall_seconds = unname(el["elapsed"]),
                cpu_seconds = unname(el["user.self"] + el["sys.self"] +
                                     el["user.child"] + el["sys.child"])), rec)
  tmp <- paste0(out, ".tmp")                        # atomic write, as in Python
  write(toJSON(rec, auto_unbox = TRUE, digits = NA), tmp)
  file.rename(tmp, out)
  cat(dataset, s$split_id, method, rec$status, "k =", rec$n_selected,
      sprintf("%.0fs", rec$wall_seconds), "\n")
}
