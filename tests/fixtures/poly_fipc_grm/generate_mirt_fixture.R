# mirt fixture generator for the unidimensional GRM FIPC agreement test
# (fixed-item calibration for polytomous GRM, MWU-MEM).
#
# What it does:
#   1. Simulates a reference group (N = 2,000, theta ~ N(0, 1)) and a focal
#      group (N = 2,000, theta ~ N(0.5, 1.2^2)) on the same 10-item,
#      3-category GRM from the model definition
#      P(Y >= k | theta) = logistic(a*theta + d_k) (Samejima, 1969) with a
#      fixed seed, and writes the focal responses to dataset_focal.csv
#      (committed).
#   2. Fits mirt(ref_anchors, itemtype = "graded") on the reference group's
#      anchor items (tight TOL so the fixed values sit at the reference MLE),
#      then runs mirt::fixedCalib(focal, old_mod) at its MWU-MEM default
#      (Kim, 2006, eqs. 14-15, pp. 361-362: prior weights updated every EM
#      cycle) and writes the fixed anchor parameters, the freely estimated
#      new-item parameters, and the observed-data log-likelihood to
#      mirt_fixture.json (committed).
#
# The Rust test fits the same focal data with the same fixed anchors and
# compares free-item slopes/intercepts and the log-likelihood. mirt updates
# a discrete empirical-histogram prior while the Rust fitter updates a
# parametric N(mu, sigma^2) prior (the Bock-Zimowski form), so agreement is
# asserted within MLE-plus-prior-family tolerance bands, not bit-exactness.
#
# Reproduce: Rscript tests/fixtures/poly_fipc_grm/generate_mirt_fixture.R
# Requires: R 4.x with mirt 1.46.1 installed.
#
# References (APA 7th ed.):
#   Kim, S. (2006). A comparative study of IRT fixed parameter calibration
#     methods. Journal of Educational Measurement, 43(4), 355-381.
#     https://doi.org/10.1111/j.1745-3984.2006.00021.x
#   Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
#     in a fixed-item linking procedure with a fixed-person prior distribution
#     for mixed-format test data. Applied Measurement in Education, 18(2),
#     199-215. https://doi.org/10.1207/s15324818ame1802_4
#   Samejima, F. (1969). Estimation of latent ability using a response pattern
#     of graded scores. Psychometrika, 34(S1), 1-97.
#     https://doi.org/10.1007/BF03372160
#   Chalmers, R. P. (2012). mirt: A multidimensional item response theory
#     package for the R environment. Journal of Statistical Software, 48(6).
#     https://doi.org/10.18637/jss.v048.i06

stopifnot(R.version$major >= 4)
stopifnot(as.character(packageVersion("mirt")) == "1.46.1")

SEED <- 20260917L
N_REF <- 2000L
N_FOC <- 2000L
N_ITEMS <- 10L
N_CAT <- 3L
N_ANCHOR <- 6L
QUADPTS <- 61L
FOCAL_MEAN <- 0.5
FOCAL_SD <- 1.2

# True GRM parameters (all slopes positive here; reverse-keyed anchors are
# covered by the Rust-side simulation tests, not by this fixture).
A <- c(1.4, 1.1, 0.9, 1.2, 1.0, 0.8, 1.3, 1.15, 0.95, 1.25)
D1 <- c(1.2, 1.0, 1.3, 0.9, 1.1, 1.4, 1.0, 1.2, 0.8, 1.1)
D2 <- c(-0.2, 0.1, -0.4, 0.0, -0.3, -0.1, -0.2, 0.2, -0.5, 0.0)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1L) {
  out_dir <- dirname(sub("^--file=", "", file_arg))
} else {
  out_dir <- file.path("tests", "fixtures", "poly_fipc_grm")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(SEED)
gen <- function(theta) {
  plogis_local <- function(x) 1 / (1 + exp(-x))
  resp <- matrix(0L, length(theta), N_ITEMS)
  for (i in seq_len(N_ITEMS)) {
    p1 <- plogis_local(A[i] * theta + D1[i])
    p2 <- plogis_local(A[i] * theta + D2[i])
    u <- runif(length(theta))
    # Categories 0..2 from the cumulative model by inversion: Y >= 1 w.p. p1
    # (u < p1), Y >= 2 w.p. p2 (u < p2); the same uniform drives both
    # indicators so the categories stay ordered.
    resp[, i] <- as.integer(u < p1) + as.integer(u < p2)
  }
  resp
}
ref <- gen(rnorm(N_REF))
foc <- gen(rnorm(N_FOC, mean = FOCAL_MEAN, sd = FOCAL_SD))
stopifnot(all(ref >= 0L & ref < N_CAT), all(foc >= 0L & foc < N_CAT))
for (i in seq_len(N_ITEMS)) {
  stopifnot(all(seq(0L, N_CAT - 1L) %in% foc[, i]))
  stopifnot(all(seq(0L, N_CAT - 1L) %in% ref[, i]))
}
colnames(ref) <- colnames(foc) <- paste0("X", seq_len(N_ITEMS))
ref_df <- data.frame(ref)
foc_df <- data.frame(foc)

dataset_path <- file.path(out_dir, "dataset_focal.csv")
write.csv(foc_df, file = dataset_path, row.names = FALSE, quote = FALSE)

# Reference calibration of the anchors (tight TOL: these values are FIXED
# downstream, so they must sit at the reference MLE, not on an EM tail).
old_mod <- mirt::mirt(
  ref_df[, seq_len(N_ANCHOR)],
  1,
  itemtype = "graded",
  quadpts = QUADPTS,
  TOL = 1e-5,
  verbose = FALSE,
  technical = list(NCYCLES = 4000L)
)
stopifnot(old_mod@OptimInfo$converged)

# FIPC at the MWU-MEM default: PAU = "MWU" updates the empirical-histogram
# prior every EM cycle (Kim, 2006, eqs. 14-15). Note: no TOL may be passed
# here — fixedCalib hardcodes TOL = NaN on its final pass, so a caller TOL
# collides as a duplicated formal.
fc <- mirt::fixedCalib(
  foc_df,
  model = 1,
  old_mod = old_mod,
  itemtype = "graded",
  quadpts = QUADPTS,
  technical = list(NCYCLES = 4000L)
)

cc_old <- mirt::coef(old_mod, simplify = TRUE)$items
cc_fc <- mirt::coef(fc, simplify = TRUE)$items
loglik_fc <- as.numeric(mirt::logLik(fc))

fixture <- list(
  seed = SEED,
  n_ref = N_REF,
  n_foc = N_FOC,
  n_items = N_ITEMS,
  n_cat = N_CAT,
  n_anchor = N_ANCHOR,
  quadpts = QUADPTS,
  focal_mean = FOCAL_MEAN,
  focal_sd = FOCAL_SD,
  true_a = A,
  true_d = cbind(D1, D2),
  anchor_a = unname(cc_old[, "a1"]),
  anchor_d = unname(cc_old[, c("d1", "d2")]),
  mirt_free_a = unname(cc_fc[(N_ANCHOR + 1L):N_ITEMS, "a1"]),
  mirt_free_d = unname(cc_fc[(N_ANCHOR + 1L):N_ITEMS, c("d1", "d2")]),
  mirt_loglik = loglik_fc,
  mirt_converged = TRUE,
  intercept_convention = paste0(
    "mirt graded d_k equals the Rust threshold: ",
    "P(Y >= k) = logistic(a*theta + d_k)"
  )
)

json_path <- file.path(out_dir, "mirt_fixture.json")
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
cat(sprintf("mirt fixedCalib logLik = %.4f\n", loglik_fc))
print(round(cc_fc, 4))
