# mirt fixture generator for the stage-4 two-tier GRM agreement test
# (stage 4 of ContextualWisdomLab/fast-mlsirm#1912).
#
# What it does:
#   1. Simulates a 10-item, 4-category two-tier graded dataset from the model
#      definition P(Y >= k) = logistic(sum_p a_ip*theta_p + a_S*theta_S + d_k)
#      with theta_P ~ MVN(0, Phi), Phi = [[1, rho], [rho, 1]] (Cai, 2010;
#      Cai, Yang, & Hansen, 2011, eq. 6-7; Gibbons et al., 2007) with a fixed
#      seed, and writes it to dataset.csv (committed).
#   2. Fits mirt::bfactor(..., model2 = <two primaries + COV>,
#      itemtype = "graded") with the SAME Gauss-Hermite quadrature density
#      the Rust test uses (quadpts per reduced dimension: ncol(G) + 1 = 3,
#      per the mirt `bfactor` documentation) and writes mirt slopes /
#      intercepts / primary correlation / log-likelihood to mirt_fixture.json
#      (committed).
#
# Reproduce: Rscript tests/fixtures/two_tier_grm_stage4/generate_mirt_fixture.R
# Requires: R 4.x with mirt 1.46.1 installed.
#
# References (APA 7th ed.):
#   Cai, L. (2010). A two-tier full-information item factor analysis model
#     with applications. Psychometrika, 75(4), 581-612.
#     https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
#     accessible — no equation locator is drawn from it)
#   Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
#     item bifactor analysis. Psychological Methods, 16(3), 221-248.
#     https://doi.org/10.1037/a0023350 (full text read: eq. 6-7)
#   Chalmers, R. P. (2026). mirt: Multidimensional item response theory
#     (Version 1.46.1) [R package]. https://cran.r-project.org/package=mirt
#     (oracle software pinned below; the `bfactor` help topic's two-tier
#     specification — Sigma = [[G, 0], [0, diag(S)]], ncol(G) + 1
#     integration dimensions — is read)

stopifnot(R.version$major >= 4)
stopifnot(as.character(packageVersion("mirt")) == "1.46.1")

SEED <- 20260917L
N_PERSONS <- 1500L
N_ITEMS <- 10L
N_CAT <- 4L
# Primary structure (simple): items 1-5 on G1, items 6-10 on G2.
PRIMARY <- c(1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L)
# Specifics cross-cut the primary split (method-factor layout):
# S1 on items 1, 2, 6, 7; S2 on items 3, 4, 5, 8, 9, 10.
SPECIFIC <- c(1L, 1L, 2L, 2L, 2L, 1L, 1L, 2L, 2L, 2L)
QUADPTS <- 15L
RHO <- 0.35

# True parameters (kept clearly identified: strong primaries, moderate
# specifics, well-separated strictly decreasing intercepts).
A_P1 <- c(1.5, 1.2, 1.0, 0.9, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0)
A_P2 <- c(0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 1.1, 1.0, 0.8, 1.2)
A_S <- c(1.0, 0.9, 1.1, 0.8, 0.7, 1.0, 1.2, 0.9, 0.7, 0.8)
D1 <- c(1.3, 1.1, 1.4, 1.0, 1.2, 1.5, 0.9, 1.1, 1.3, 1.0)
D2 <- c(0.1, -0.1, 0.2, 0.0, 0.1, 0.3, -0.2, 0.0, 0.2, -0.1)
D3 <- c(-1.1, -1.3, -1.0, -1.2, -1.1, -0.9, -1.4, -1.2, -1.0, -1.3)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1L) {
  out_dir <- dirname(sub("^--file=", "", file_arg))
} else {
  # Fallback when sourced without a file path: assume repo-root CWD.
  out_dir <- file.path("tests", "fixtures", "two_tier_grm_stage4")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(SEED)
z0 <- rnorm(N_PERSONS)
z1 <- rnorm(N_PERSONS)
theta_g1 <- z0
theta_g2 <- RHO * z0 + sqrt(1 - RHO^2) * z1
theta_s <- matrix(rnorm(N_PERSONS * 2L), N_PERSONS, 2L)

plogis <- function(x) 1 / (1 + exp(-x))
resp <- matrix(0L, N_PERSONS, N_ITEMS)
for (i in seq_len(N_ITEMS)) {
  s <- SPECIFIC[i]
  base <- A_P1[i] * theta_g1 + A_P2[i] * theta_g2 + A_S[i] * theta_s[, s]
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

# mirt models graded responses 0..K-1 with itemtype = "graded". The `model`
# vector assigns each item to its specific factor; `model2` assigns the
# primary (first-tier) structure with COV freeing the G1-G2 correlation
# (mirt `bfactor` documentation: the two-tier covariance is
# Sigma = [[G, 0], [0, diag(S)]] with primaries free to vary/covary and
# specifics orthogonal with unit variance). mirt's graded slope-intercept
# form is P(Y >= k | theta) = logistic(sum_p a_p*theta_p + a_S*theta_S + d_k),
# i.e. its reported `d` columns are the SAME additive boundary intercepts
# the Rust fitter calls `threshold` (no sign flip, no a-scaling). The
# agreement test documents this mapping and compares directly.
model2 <- mirt::mirt.model("G1 = 1-5
                             G2 = 6-10
                             COV = G1*G2")
mod <- mirt::bfactor(
  data.frame(resp),
  model = SPECIFIC,
  model2 = model2,
  itemtype = "graded",
  quadpts = QUADPTS,
  TOL = 5e-4,
  technical = list(NCYCLES = 4000L)
)
stopifnot(mod@OptimInfo$converged)

cc <- mirt::coef(mod, simplify = TRUE)$items
# Columns: a1/a2 (primaries G1/G2, 0 off-pattern), a3/a4 (specifics S1/S2,
# 0 off-block), d1..d3 (intercepts).
a_p1_mirt <- unname(cc[, "a1"])
a_p2_mirt <- unname(cc[, "a2"])
a_s_mirt <- sapply(seq_len(N_ITEMS), function(i) {
  s <- SPECIFIC[i]
  unname(cc[i, paste0("a", 2L + s)])
})
d_mirt <- unname(cc[, c("d1", "d2", "d3")])
phi_mirt <- unname(mirt::coef(mod, simplify = TRUE)$cov[1:2, 1:2])
loglik_mirt <- as.numeric(mirt::logLik(mod))

fixture <- list(
  seed = SEED,
  n_persons = N_PERSONS,
  n_items = N_ITEMS,
  n_cat = N_CAT,
  n_primary = 2L,
  primary = PRIMARY,
  specific = SPECIFIC,
  quadpts = QUADPTS,
  true_rho = RHO,
  true_a_primary = cbind(A_P1, A_P2),
  true_a_specific = A_S,
  true_d = cbind(D1, D2, D3),
  mirt_a_primary = cbind(a_p1_mirt, a_p2_mirt),
  mirt_a_specific = a_s_mirt,
  mirt_d = d_mirt,
  mirt_phi = phi_mirt,
  mirt_loglik = loglik_mirt,
  mirt_converged = TRUE,
  intercept_convention = paste0(
    "mirt graded d_k equals the Rust threshold: ",
    "P(Y >= k) = logistic(sum_p a_ip*theta_p + a_S*theta_S + d_k)"
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
print(round(phi_mirt, 4))
