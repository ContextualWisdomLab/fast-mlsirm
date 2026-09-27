# Reproduce from repo root:
#   Rscript tests/fixtures/two_tier_grm_oakes/generate_mirt_oakes_fixture.R
# Requires: R 4.x with mirt 1.46.1
#
# Data: stage-4 dataset.csv (seed 20260917, 1500 persons x 10 graded items).
# Model: mirt::bfactor with model2 G1=1-5 / G2=6-10 / COV=G1*G2,
#        SE.type="Oakes", quadpts=15 (implementation cross-check; not study
#        settings — study settings use >= 121 nodes/dim).
#
# Paper basis: Oakes (1999, eq. 6); Cai et al. (2011, eq. 6-7); Cai (2010)
# abstract (two-tier model; full text not accessible).

stopifnot(R.version$major >= 4)
stopifnot(as.character(packageVersion("mirt")) == "1.46.1")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) == 1L) {
  out_dir <- dirname(sub("^--file=", "", file_arg))
} else {
  out_dir <- file.path("tests", "fixtures", "two_tier_grm_oakes")
}
stage4 <- file.path(dirname(out_dir), "two_tier_grm_stage4")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
resp <- read.csv(file.path(stage4, "dataset.csv"))
N_ITEMS <- ncol(resp)
SPECIFIC <- c(1L, 1L, 2L, 2L, 2L, 1L, 1L, 2L, 2L, 2L)
QUADPTS <- 15L
model2 <- mirt::mirt.model("G1 = 1-5
                             G2 = 6-10
                             COV = G1*G2")
mod <- mirt::bfactor(
  resp, model = SPECIFIC, model2 = model2, itemtype = "graded",
  quadpts = QUADPTS, TOL = 5e-4, SE = TRUE, SE.type = "Oakes",
  technical = list(NCYCLES = 4000L)
)
stopifnot(isTRUE(mod@OptimInfo$converged))
cc <- mirt::coef(mod, simplify = TRUE)$items
a_p1 <- unname(cc[, "a1"]); a_p2 <- unname(cc[, "a2"])
a_s <- sapply(seq_len(N_ITEMS), function(i) unname(cc[i, paste0("a", 2L + SPECIFIC[i])]))
d <- unname(cc[, c("d1", "d2", "d3")])
phi <- unname(mirt::coef(mod, simplify = TRUE)$cov[1:2, 1:2])
V <- mod@vcov
rn <- rownames(V)
se_all <- sqrt(diag(V))
con <- file(file.path(out_dir, "mirt_oakes_fixture.json"), open = "wt")
cat("{\n", file = con)
cat('  "dataset": "../two_tier_grm_stage4/dataset.csv",\n', file = con)
cat(sprintf('  "seed": 20260917,\n'), file = con)
cat(sprintf('  "n_persons": %d,\n', nrow(resp)), file = con)
cat(sprintf('  "n_items": %d,\n', N_ITEMS), file = con)
cat(sprintf('  "n_cat": 4,\n'), file = con)
cat(sprintf('  "n_primary": 2,\n'), file = con)
cat('  "primary": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],\n', file = con)
cat(sprintf('  "specific": [%s],\n', paste(SPECIFIC, collapse = ", ")), file = con)
cat(sprintf('  "quadpts": %d,\n', QUADPTS), file = con)
cat(sprintf('  "mirt_version": "%s",\n', packageVersion("mirt")), file = con)
cat(sprintf('  "se_type": "Oakes",\n'), file = con)
cat(sprintf('  "mirt_a_primary": [\n'), file = con)
for (i in seq_len(N_ITEMS)) {
  cat(sprintf('    [%s, %s]%s\n', format(a_p1[i], digits=15, scientific=FALSE), format(a_p2[i], digits=15, scientific=FALSE), if (i<N_ITEMS) "," else ""), file = con)
}
cat('  ],\n', file = con)
cat(sprintf('  "mirt_a_specific": [%s],\n', paste(format(a_s, digits=15, scientific=FALSE), collapse=", ")), file = con)
cat('  "mirt_d": [\n', file = con)
for (i in seq_len(N_ITEMS)) {
  cat(sprintf('    [%s, %s, %s]%s\n', format(d[i,1], digits=15, scientific=FALSE), format(d[i,2], digits=15, scientific=FALSE), format(d[i,3], digits=15, scientific=FALSE), if (i<N_ITEMS) "," else ""), file = con)
}
cat('  ],\n', file = con)
cat(sprintf('  "mirt_phi": [[%s, %s], [%s, %s]],\n', format(phi[1,1], digits=15, scientific=FALSE), format(phi[1,2], digits=15, scientific=FALSE), format(phi[2,1], digits=15, scientific=FALSE), format(phi[2,2], digits=15, scientific=FALSE)), file = con)
cat(sprintf('  "mirt_loglik": %s,\n', format(as.numeric(mirt::logLik(mod)), digits=15, scientific=FALSE)), file = con)
cat(sprintf('  "mirt_vcov_dimnames": [%s],\n', paste(sprintf('"%s"', rn), collapse=", ")), file = con)
cat('  "mirt_se": [', paste(format(se_all, digits=15, scientific=FALSE), collapse=", "), '],\n', file = con, sep="")
cat('  "mirt_vcov": [\n', file = con)
for (r in seq_len(nrow(V))) {
  cat('    [', paste(format(V[r,], digits=15, scientific=FALSE), collapse=", "), ']', if (r<nrow(V)) "," else "", '\n', file = con, sep="")
}
cat('  ],\n', file = con)
cat('  "mirt_converged": true\n}\n', file = con)
close(con)
cat("Wrote", file.path(out_dir, "mirt_oakes_fixture.json"), "k=", length(se_all), "\n")
