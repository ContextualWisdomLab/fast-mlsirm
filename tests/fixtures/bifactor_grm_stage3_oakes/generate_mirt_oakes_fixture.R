# mirt Oakes-SE fixture generator for the stage-3 Oakes test
# (stage 3 of ContextualWisdomLab/fast-mlsirm#1912).
#
# What it does: reads the STAGE-1 fixture dataset
# (tests/fixtures/bifactor_grm_stage1/dataset.csv — the SAME data the
# stage-1 mirt-agreement test uses, so the Rust-vs-mirt SE comparison stays
# on identical data), fits mirt::bfactor(..., itemtype = "graded") with
# SE=TRUE, SE.type="Oakes" at the SAME quadrature density the Rust test uses
# (quadpts per integrated dimension), and writes mirt slopes / intercepts /
# Oakes SEs / the full 40x40 vcov (with dimnames) / log-likelihood to
# mirt_oakes_fixture.json (committed).
#
# Orientation note: stage-1 generator historically used `u > p_k` (category-
# reversed). #1950 fixed it to `u < p_k` and regenerated dataset.csv /
# mirt_fixture.json. This Oakes fixture must be regenerated against that
# corrected dataset whenever the stage-1 CSV changes — otherwise Rust
# evaluates Oakes at a foreign MLE and the observed information need not
# be PD (the #2011 CI failure mode).
#
# Intercept mapping (same as the stage-1 agreement test, verified against
# the mirt fit, not assumed): mirt's graded `d_k` columns are the SAME
# additive boundary intercepts the Rust fitter reports as `threshold`, i.e.
# `P(Y >= k | theta) = logistic(a_G*theta_G + a_S*theta_S + d_k)`.
#
# GRID SCOPE (maintainer quadrature rule): quadpts = 15 is a matched small
# grid for an implementation cross-check (both sides share the grid family),
# NOT study settings. The 121+-node study-settings fixture follows the
# SUPPORTED_Q cap removal (rebase on fix/1929-quadrature-defaults).
#
# Reproduce: Rscript tests/fixtures/bifactor_grm_stage3_oakes/generate_mirt_oakes_fixture.R
# Requires: R 4.x with mirt 1.46.1 installed.
#
# Model basis (full text read):
#   Gibbons, R. D., et al. (2007). Full-information item bifactor analysis of
#     graded response data. Applied Psychological Measurement, 31(1), 4-19.
#     https://doi.org/10.1177/0146621606289485
# Oracle provenance (factual): R 4.x, mirt 1.46.1, SE.type="Oakes".

stopifnot(R.version$major >= 4)
stopifnot(as.character(packageVersion("mirt")) == "1.46.1")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1L) {
  out_dir <- dirname(sub("^--file=", "", file_arg))
} else {
  out_dir <- file.path("tests", "fixtures", "bifactor_grm_stage3_oakes")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
stage1_dir <- file.path(dirname(out_dir), "bifactor_grm_stage1")

resp <- read.csv(file.path(stage1_dir, "dataset.csv"))
N_ITEMS <- ncol(resp)
# Items 1-4 on specific 1, items 5-8 on specific 2 (stage-1 layout).
SPECIFIC <- c(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L)
QUADPTS <- 15L

# mirt's TOL gates its EM loglik slope (see the stage-1 generator notes).
mod <- mirt::bfactor(
  resp,
  model = SPECIFIC,
  itemtype = "graded",
  quadpts = QUADPTS,
  TOL = 5e-4,
  SE = TRUE,
  SE.type = "Oakes",
  technical = list(NCYCLES = 4000L)
)
stopifnot(isTRUE(mod@OptimInfo$converged))

cc <- mirt::coef(mod, simplify = TRUE)$items
a_g_mirt <- unname(cc[, "a1"])
a_s_mirt <- sapply(seq_len(N_ITEMS), function(i) {
  s <- SPECIFIC[i]
  unname(cc[i, paste0("a", 1L + s)])
})
d_mirt <- unname(cc[, c("d1", "d2", "d3")])
loglik_mirt <- as.numeric(mirt::logLik(mod))

# vcov alignment: rownames are "par.flatidx" where flatidx runs over mirt's
# internal 6-slot item blocks (a1, a2, a3, d1, d2, d3); fixed off-block
# slopes have no row. Rebuild (item, par) per row and verify against the
# coef table layout.
V <- mod@vcov
stopifnot(is.matrix(V), nrow(V) == 5L * N_ITEMS, ncol(V) == 5L * N_ITEMS)
rn <- rownames(V)
stopifnot(!is.null(rn), length(rn) == 5L * N_ITEMS)
row_item <- integer(length(rn))
row_par <- character(length(rn))
for (r in seq_along(rn)) {
  parts <- strsplit(rn[r], ".", fixed = TRUE)[[1L]]
  stopifnot(length(parts) == 2L)
  par <- parts[1L]
  flat <- as.integer(parts[2L])
  item <- (flat - 1L) %/% 6L + 1L
  slot <- (flat - 1L) %% 6L + 1L
  expect_slot <- switch(par,
    a1 = 1L, a2 = 2L, a3 = 3L, d1 = 4L, d2 = 5L, d3 = 6L,
    stop(paste("unexpected vcov par name", par)))
  stopifnot(slot == expect_slot, item >= 1L, item <= N_ITEMS)
  # Off-block specific slopes are fixed (no vcov row); check consistency.
  if (par %in% c("a2", "a3")) {
    stopifnot(as.integer(sub("a", "", par)) == 1L + SPECIFIC[item])
  }
  row_item[r] <- item
  row_par[r] <- par
}
se_all <- sqrt(diag(V))
se_of <- function(item, par) {
  idx <- which(row_item == item & row_par == par)
  stopifnot(length(idx) == 1L)
  unname(se_all[idx])
}
se_a_g <- sapply(seq_len(N_ITEMS), function(i) se_of(i, "a1"))
se_a_s <- sapply(seq_len(N_ITEMS), function(i) {
  se_of(i, paste0("a", 1L + SPECIFIC[i]))
})
se_d <- t(sapply(seq_len(N_ITEMS), function(i) {
  c(se_of(i, "d1"), se_of(i, "d2"), se_of(i, "d3"))
}))

fixture <- list(
  dataset = "../bifactor_grm_stage1/dataset.csv",
  seed = 20260916L,
  n_persons = nrow(resp),
  n_items = N_ITEMS,
  n_cat = 4L,
  specific = SPECIFIC,
  quadpts = QUADPTS,
  mirt_version = as.character(packageVersion("mirt")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  se_type = "Oakes",
  tol = 5e-4,
  mirt_a_general = a_g_mirt,
  mirt_a_specific = a_s_mirt,
  mirt_d = d_mirt,
  mirt_se_a_general = se_a_g,
  mirt_se_a_specific = se_a_s,
  mirt_se_d = se_d,
  mirt_vcov = V,
  mirt_vcov_dimnames = rn,
  mirt_vcov_order_item = row_item,
  mirt_vcov_order_par = row_par,
  mirt_loglik = loglik_mirt,
  mirt_converged = TRUE,
  intercept_convention = paste0(
    "mirt graded d_k equals the Rust threshold: ",
    "P(Y >= k) = logistic(a_G*theta_G + a_S*theta_S + d_k)"
  )
)
json_path <- file.path(out_dir, "mirt_oakes_fixture.json")
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
      cat(pad, "  [", paste(format(x[r, ], scientific = FALSE, digits = 15L), collapse = ", "),
        "]", sep = "", file = con)
      if (r < nrow(x)) cat(",", file = con)
      cat("\n", file = con, sep = "")
    }
    cat(pad, "]", file = con, sep = "")
  } else if (length(x) > 1L) {
    if (is.character(x)) {
      quoted <- vapply(x, function(s) {
        paste0("\"", gsub("\\", "\\\\", gsub("\"", "\\\"", s, fixed = TRUE), fixed = TRUE), "\"")
      }, character(1L), USE.NAMES = FALSE)
      cat("[", paste(quoted, collapse = ", "), "]", sep = "", file = con)
    } else {
      cat("[", paste(format(x, scientific = FALSE, digits = 15L), collapse = ", "), "]",
        sep = "", file = con)
    }
  } else if (is.character(x)) {
    cat("\"", gsub("\"", "\\\"", x, fixed = TRUE), "\"", sep = "", file = con)
  } else if (is.logical(x)) {
    cat(tolower(as.character(x)), file = con, sep = "")
  } else {
    cat(format(x, scientific = FALSE, digits = 15L), file = con, sep = "")
  }
}
con <- file(json_path, open = "wt")
write_json_value(con, fixture, 0L)
cat("\n", file = con)
close(con)

cat("wrote", json_path, "\n")
cat(sprintf("mirt logLik = %.4f\n", loglik_mirt))
cat("mirt SEs (a_general):\n")
print(round(se_a_g, 4))
