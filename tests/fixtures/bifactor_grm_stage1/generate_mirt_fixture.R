# mirt fixture generator for the stage-1 bifactor GRM agreement test
# (stage 1 of ContextualWisdomLab/fast-mlsirm#1912).
#
# What it does:
#   1. Simulates an 8-item, 4-category bifactor graded dataset from the model
#      definition P(Y >= k) = logistic(a_G*theta_G + a_S*theta_S + d_k)
#      (Gibbons et al., 2007; Gibbons & Hedeker, 1992; Samejima, 1969) with
#      a fixed seed, and writes it to dataset.csv (committed).
#   2. Fits mirt::bfactor(..., itemtype = "graded") with the SAME Gauss-Hermite
#      quadrature density the Rust test uses (quadpts per integrated
#      dimension) and writes mirt slopes / intercepts / log-likelihood to
#      mirt_fixture.json (committed).
#
# Reproduce: Rscript tests/fixtures/bifactor_grm_stage1/generate_mirt_fixture.R
# Requires: R 4.x with mirt 1.46.1 installed.
#
# References (APA 7th ed.):
#   Gibbons, R. D., et al. (2007). Full-information item bifactor analysis of
#     graded response data. Applied Psychological Measurement, 31(1), 4-19.
#     https://doi.org/10.1177/0146621606289485
#   Chalmers, R. P. (2012). mirt: A multidimensional item response theory
#     package for the R environment. Journal of Statistical Software, 48(6).
#     https://doi.org/10.18637/jss.v048.i06

stopifnot(R.version$major >= 4)
stopifnot(as.character(packageVersion("mirt")) == "1.46.1")

SEED <- 20260916L
N_PERSONS <- 800L
N_ITEMS <- 8L
N_CAT <- 4L
# Items 1-4 on specific 1, items 5-8 on specific 2.
SPECIFIC <- c(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L)
QUADPTS <- 15L

# True parameters (kept clearly identified: strong general, moderate
# specifics, well-separated strictly decreasing intercepts).
A_G <- c(1.5, 1.2, 1.0, 0.9, 1.4, 1.1, 1.0, 0.8)
A_S <- c(1.0, 0.9, 1.1, 0.8, 1.0, 1.2, 0.9, 0.7)
D1 <- c(1.3, 1.1, 1.4, 1.0, 1.2, 1.5, 0.9, 1.1)
D2 <- c(0.1, -0.1, 0.2, 0.0, 0.1, 0.3, -0.2, 0.0)
D3 <- c(-1.1, -1.3, -1.0, -1.2, -1.1, -0.9, -1.4, -1.2)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1L) {
  out_dir <- dirname(sub("^--file=", "", file_arg))
} else {
  # Fallback when sourced without a file path: assume repo-root CWD.
  out_dir <- file.path("tests", "fixtures", "bifactor_grm_stage1")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(SEED)
theta_g <- rnorm(N_PERSONS)
theta_s <- matrix(rnorm(N_PERSONS * 2L), N_PERSONS, 2L)

plogis <- function(x) 1 / (1 + exp(-x))
resp <- matrix(0L, N_PERSONS, N_ITEMS)
for (i in seq_len(N_ITEMS)) {
  s <- SPECIFIC[i]
  base <- A_G[i] * theta_g + A_S[i] * theta_s[, s]
  p1 <- plogis(base + D1[i])
  p2 <- plogis(base + D2[i])
  p3 <- plogis(base + D3[i])
  u <- runif(N_PERSONS)
  # Categories 0..3 from the cumulative model: Y >= 1 w.p. p1, etc.
  resp[, i] <- as.integer(u > p1) + as.integer(u > p2) + as.integer(u > p3)
}
stopifnot(all(resp >= 0L & resp < N_CAT))
# Every declared category must be observed for every item (the Rust fitter
# refuses unidentified boundaries instead of imputing them).
for (i in seq_len(N_ITEMS)) {
  stopifnot(all(seq(0L, N_CAT - 1L) %in% resp[, i]))
}

dataset_path <- file.path(out_dir, "dataset.csv")
write.csv(
  data.frame(resp),
  file = dataset_path, row.names = FALSE, quote = FALSE
)

# mirt models graded responses 0..K-1 with itemtype = "graded". The bfactor
# `model` vector assigns each item to its specific factor; the general factor
# is implicit. mirt's graded slope-intercept form is
#   P(Y >= k | theta) = logistic(a_G*theta_G + a_S*theta_S + d_k),
# i.e. its reported `d` columns are the SAME additive boundary intercepts the
# Rust fitter calls `threshold` (no sign flip, no a-scaling). The agreement
# test documents this mapping and compares directly.
# mirt's TOL gates its EM loglik slope: TOL = 5e-4 converges in ~230 cycles
# (logLik moves < 0.03 vs TOL = 1e-3), while TOL = 1e-5 does not terminate
# within 2000 cycles on this data (diminishing EM tail, not a different
# optimum). The Rust side runs its own tolerance; agreement is on the MLE.
mod <- mirt::bfactor(
  data.frame(resp),
  model = SPECIFIC,
  itemtype = "graded",
  quadpts = QUADPTS,
  TOL = 5e-4,
  technical = list(NCYCLES = 4000L)
)
stopifnot(mod@OptimInfo$converged)

cc <- mirt::coef(mod, simplify = TRUE)$items
# Columns: a1 (general), a2/a3 (specifics, 0 off-block), d1..d3 (intercepts).
a_g_mirt <- unname(cc[, "a1"])
a_s_mirt <- sapply(seq_len(N_ITEMS), function(i) {
  s <- SPECIFIC[i]
  unname(cc[i, paste0("a", 1L + s)])
})
d_mirt <- unname(cc[, c("d1", "d2", "d3")])
loglik_mirt <- as.numeric(mirt::logLik(mod))

fixture <- list(
  seed = SEED,
  n_persons = N_PERSONS,
  n_items = N_ITEMS,
  n_cat = N_CAT,
  specific = SPECIFIC,
  quadpts = QUADPTS,
  true_a_general = A_G,
  true_a_specific = A_S,
  true_d = cbind(D1, D2, D3),
  mirt_a_general = a_g_mirt,
  mirt_a_specific = a_s_mirt,
  mirt_d = d_mirt,
  mirt_loglik = loglik_mirt,
  mirt_converged = TRUE,
  intercept_convention = paste0(
    "mirt graded d_k equals the Rust threshold: ",
    "P(Y >= k) = logistic(a_G*theta_G + a_S*theta_S + d_k)"
  )
)
json_path <- file.path(out_dir, "mirt_fixture.json")
# Minimal JSON writer (no extra package dependency).
write_json_value <- function(con, x, indent) {
  pad <- paste(rep("  ", indent), collapse = "")
  if (is.list(x) && is.null(names(x))) {
    cat("[\n", file = con, sep = "")
    for (i in seq_along(x)) {
      cat(pad, "  ", sep = "", file = con)
      write_json_value(con, x[[i]], indent + 1L)
      if (i < length(x)) cat(",", file = con)
      cat("\n", file = con, sep = "")
    }
    cat(pad, "]", file = con, sep = "")
  } else if (is.list(x)) {
    cat("{\n", file = con, sep = "")
    nms <- names(x)
    for (i in seq_along(x)) {
      cat(pad, "  \"", nms[i], "\": ", sep = "", file = con)
      write_json_value(con, x[[i]], indent + 1L)
      if (i < length(x)) cat(",", file = con)
      cat("\n", file = con, sep = "")
    }
    cat(pad, "}", file = con, sep = "")
  } else if (is.matrix(x)) {
    cat("[\n", file = con, sep = "")
    for (r in seq_len(nrow(x))) {
      cat(pad, "  [", paste(format(x[r, ], scientific = FALSE), collapse = ", "),
        "]", sep = "", file = con)
      if (r < nrow(x)) cat(",", file = con)
      cat("\n", file = con, sep = "")
    }
    cat(pad, "]", file = con, sep = "")
  } else if (length(x) > 1L) {
    cat("[", paste(format(x, scientific = FALSE), collapse = ", "), "]",
      sep = "", file = con)
  } else if (is.character(x)) {
    cat("\"", gsub("\"", "\\\"", x, fixed = TRUE), "\"", sep = "", file = con)
  } else if (is.logical(x)) {
    cat(tolower(as.character(x)), file = con, sep = "")
  } else {
    cat(format(x, scientific = FALSE), file = con, sep = "")
  }
}
con <- file(json_path, open = "wt")
write_json_value(con, fixture, 0L)
cat("\n", file = con)
close(con)

cat("wrote", dataset_path, "and", json_path, "\n")
cat(sprintf("mirt logLik = %.4f\n", loglik_mirt))
print(round(cc, 4))
