//! Block-level response partial-pattern index for bifactor / two-tier E-steps.
//!
//! # Why this exists (#2003)
//!
//! The reduced bifactor person marginal factors per general node as
//! `L_p = sum_g w_g G_pg prod_s I_psg` with
//! `I_psg = sum_h v_h prod_{i in block_s} P(Y_pi | g, h)` (Gibbons et al.,
//! 2007, eq. 15). For fixed item parameters, `I_psg` (and the underlying
//! `(g, h)` accumulators) depend on person `p` **only through the response
//! partial pattern on block `s`**, including which members are missing.
//! Persons who share that partial pattern therefore share identical block
//! contributions and need not recompute them.
//!
//! # Paper basis (APA 7th; verified locators)
//!
//! Bock and Aitkin (1981, p. 445) write that observed pattern counts assign
//! each subject to one multinomial category and that
//! `log L = C + Σ_l r_l log P_l`; on p. 448 they recode subjects to distinct
//! score patterns and note there are `s` such patterns (and hence `s` values
//! of `E(θ|x_l)`). That is whole-pattern frequency EM.
//!
//! Applying the same collapse **per specific-factor block** requires the
//! bifactor conditional-independence / product factorization: Gibbons and
//! Hedeker (1992, p. 423) state items are "conditionally independent between
//! paragraphs, but conditionally dependent within paragraphs," and (p. 425)
//! that the bifactor restriction reduces the `s`-fold integral to a
//! two-dimensional integral whose specific-dimension contributions multiply.
//! Gibbons et al. (2007, p. 8) state that the bifactor restriction always
//! results in a two-dimensional integral regardless of the number of
//! dimensions. Gibbons et al. (2007, p. 9) state that the integral stays
//! two-dimensional for both the binary and the graded bifactor models
//! regardless of the number of subdomains. Together those two results make
//! a per-block sub-pattern collapse an exact transformation of the reduced
//! integral, not an approximation. See
//! `docs/papers/2003-block-partial-pattern-source-check.md`.
//!
//! # References
//!
//! Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
//! of item parameters: Application of an EM algorithm. *Psychometrika,
//! 46*(4), 443–459. https://doi.org/10.1007/BF02293801
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423–436.
//! https://doi.org/10.1007/BF02295430
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
//! Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
//! Stover, A. (2007). Full-information item bifactor analysis of graded
//! response data. *Applied Psychological Measurement, 31*(1), 4–19.
//! https://doi.org/10.1177/0146621606289485

use std::collections::HashMap;

/// Sentinel byte for a missing response inside a partial-pattern key.
/// Distinct from every observed category `0..n_cat-1` (requires `n_cat <= 254`).
pub(crate) const PATTERN_MISSING: u8 = 0xff;

/// Measured before/after unique-pattern counts (ADR-0028: no unsourced defaults).
///
/// Every field is taken from the response matrix and block membership; nothing
/// is clamped or filled with a placeholder reduction factor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockPatternCollapseProvenance {
    /// Person count `N` used as the pre-collapse unit for every block.
    pub n_persons: usize,
    /// Pre-collapse count per specific block (always `n_persons` each).
    pub n_persons_per_block: Vec<usize>,
    /// Post-collapse unique partial-pattern counts per specific block.
    pub n_unique_patterns_per_block: Vec<usize>,
    /// Pre-collapse count for the general-only item set (`n_persons`).
    pub n_persons_general_only: usize,
    /// Unique partial patterns on the general-only item set (missingness included).
    pub n_unique_general_only_patterns: usize,
}

impl BlockPatternCollapseProvenance {
    /// Total unique block partial patterns across specifics (excludes general-only).
    pub fn total_unique_block_patterns(&self) -> usize {
        self.n_unique_patterns_per_block.iter().sum()
    }
}

/// One block's unique partial patterns and the person → pattern map.
#[derive(Clone, Debug)]
pub(crate) struct BlockPatternTable {
    /// `patterns[k][m]` = category or [`PATTERN_MISSING`] for member `m`.
    pub patterns: Vec<Vec<u8>>,
    /// `person_to_pattern[p]` = pattern index `k` in `0..patterns.len()`.
    pub person_to_pattern: Vec<usize>,
}

/// Full index over specific blocks plus the general-only item set.
#[derive(Clone, Debug)]
pub(crate) struct BlockPartialPatternIndex {
    pub blocks: Vec<BlockPatternTable>,
    pub general_only: BlockPatternTable,
    pub provenance: BlockPatternCollapseProvenance,
}

fn encode_partial(
    y: &[usize],
    observed: Option<&[bool]>,
    n_items: usize,
    n_cat: usize,
    members: &[usize],
    person: usize,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(members.len());
    for &i in members {
        let obs = observed.is_none_or(|o| o[person * n_items + i]);
        if !obs {
            key.push(PATTERN_MISSING);
        } else {
            let yc = y[person * n_items + i];
            debug_assert!(yc < n_cat && yc < PATTERN_MISSING as usize);
            key.push(yc as u8);
        }
    }
    key
}

fn build_table(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    members: &[usize],
) -> BlockPatternTable {
    let mut map: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut patterns: Vec<Vec<u8>> = Vec::new();
    let mut person_to_pattern = Vec::with_capacity(n_persons);
    for p in 0..n_persons {
        let key = encode_partial(y, observed, n_items, n_cat, members, p);
        let idx = if let Some(&existing) = map.get(&key) {
            existing
        } else {
            let idx = patterns.len();
            map.insert(key.clone(), idx);
            patterns.push(key);
            idx
        };
        person_to_pattern.push(idx);
    }
    BlockPatternTable {
        patterns,
        person_to_pattern,
    }
}

/// Build the block partial-pattern index once per fit (data-fixed).
///
/// Returns `Err` when `n_cat > 254` (sentinel collision with [`PATTERN_MISSING`]).
pub(crate) fn build_block_partial_pattern_index(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    blocks: &[Vec<usize>],
    general_only: &[usize],
) -> Result<BlockPartialPatternIndex, String> {
    if n_cat > PATTERN_MISSING as usize {
        return Err(format!(
            "n_cat={n_cat} exceeds the block-pattern missing sentinel ({PATTERN_MISSING}); \
             refuse to encode partial patterns"
        ));
    }
    let mut block_tables = Vec::with_capacity(blocks.len());
    let mut n_unique = Vec::with_capacity(blocks.len());
    let mut n_before = Vec::with_capacity(blocks.len());
    for members in blocks {
        let table = build_table(y, observed, n_persons, n_items, n_cat, members);
        n_unique.push(table.patterns.len());
        n_before.push(n_persons);
        block_tables.push(table);
    }
    let gen_table = build_table(y, observed, n_persons, n_items, n_cat, general_only);
    let provenance = BlockPatternCollapseProvenance {
        n_persons,
        n_persons_per_block: n_before,
        n_unique_patterns_per_block: n_unique,
        n_persons_general_only: n_persons,
        n_unique_general_only_patterns: gen_table.patterns.len(),
    };
    Ok(BlockPartialPatternIndex {
        blocks: block_tables,
        general_only: gen_table,
        provenance,
    })
}

/// Public measurement of collapse provenance for callers/tests (ADR-0028).
pub fn block_pattern_collapse_provenance(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    blocks: &[Vec<usize>],
    general_only: &[usize],
) -> Result<BlockPatternCollapseProvenance, String> {
    Ok(
        build_block_partial_pattern_index(
            y,
            observed,
            n_persons,
            n_items,
            n_cat,
            blocks,
            general_only,
        )?
        .provenance,
    )
}
