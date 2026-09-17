#!/usr/bin/env python3
"""Apply ADR-0028's naming/defaults policy to an inventory CSV.

Reads ``docs/api/inventory-<date>.csv`` (from ``tools/inventory_public_api.py``)
and writes ``docs/api/renames-and-defaults-<date>.csv``: one row per
(callable, defaulted-parameter) pair recording the ADR-0028 naming decision
for the callable and the default-argument decision for the parameter.
Callables with no defaulted parameters get a single row with the parameter
columns empty.

This is a mechanical, rule-based classification, not a per-function research
pass (ADR-0028, "Non-goals"): most numerical-precision/threshold/seed/model
defaults come out ``require`` because no in-repo citation for the specific
value exists today. Sourcing a defensible value, or overriding an edge-case
misclassification, is deferred to the per-module implementation sub-issues.

Usage:
    python tools/classify_renames_and_defaults.py --date 20260917
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FIELDS = [
    "source",
    "current_name",
    "proposed_name",
    "name_decision",
    "name_rationale",
    "module",
    "kind",
    "parameter",
    "current_default",
    "default_decision",
    "default_rationale",
    "migration_note",
]

# --- naming: known ADR-0028 violations -> corrected name -------------------
# Built from the positional-majority analysis in ADR-0028 (docs/adr/0028-*).
RENAME_MAP = {
    # polytomous: majority is a trailing scope qualifier.
    "polytomous_expected_response": "predict_polytomous_expected_response",
    "polytomous_category_probabilities": "predict_polytomous_category_probabilities",
    "polytomous_information_criteria": "score_polytomous_information_criteria",
    "dif_polytomous_anchor_sets": "detect_dif_polytomous_anchor_sets",
    "dif_polytomous_purified": "detect_dif_polytomous_purified",
    "dif_polytomous": "detect_dif_polytomous",
    # bifactor: majority is verb-adjacent prefix, not trailing.
    "direct_enumeration_bifactor": "enumerate_bifactor_direct",
    "focal_expected_total_score_monotonicity": "bifactor_focal_expected_total_score_monotonicity",
    # DIF family: unify on the detect_dif_<method> prefix form.
    "mantel_haenszel_dif": "detect_dif_mantel_haenszel",
    "gmh_dif": "detect_dif_gmh",
    "breslow_day_dif": "detect_dif_breslow_day",
    "logistic_dif": "detect_dif_logistic",
    "mantel_haenszel_dif_purified": "detect_dif_mantel_haenszel_purified",
    "logistic_dif_purified": "detect_dif_logistic_purified",
    "dif_analysis": "detect_dif_summary",
    # verb-first violations (qualifier before verb).
    "cat_simulate_polytomous": "simulate_cat_polytomous",
    "cat_next_item": "select_cat_next_item",
}
NAME_RATIONALE = (
    "ADR-0028 naming convention: model/family token is a prefix immediately "
    "after the verb, scope qualifier (data shape) is a trailing suffix, "
    "verb always leads."
)

# --- defaults: parameter-name -> policy family ------------------------------
QUADRATURE_RE = re.compile(
    r"^(q_theta|q_xi|q_u|q_gamma|n_quad|n_nodes|xi_points|m2_q_theta|m2_q_u|"
    r"m2_q_xi|nevalpoints|node_dims)$"
)
CONVERGENCE_RE = re.compile(
    r"^(max_iter|tol|tolerance|max_cycles|max_line_search|m_steps|mh_steps|"
    r"n_iterations|burn_in|target_accept|basin_tolerance|omega_tol|"
    r"sig2_stop|hessian_step|eps|eps_distance|epsilon)$"
)
REPLICATES_RE = re.compile(
    r"^(n_rep|n_replicates|n_boot|n_draws|n_starts|n_sample_splits|k_folds|"
    r"init_games)$"
)
SEED_RE = re.compile(r"seed")
THRESHOLD_RE = re.compile(
    r"^(alpha|alpha_level|fdr_q|se_threshold|flag_threshold|min_expected|"
    r"min_discrimination|min_effect|person_flag_threshold|"
    r"itemfit_penalty_weight|msq_band|isolation_z|sx2_min_effect|"
    r"zero_tolerance|max_rounds|min_anchor_items|min_flags_to_remove|j_min)$"
)
MODEL_RE = re.compile(r"^model$")

PARAM_RE = re.compile(r"(\w+)\s*=\s*((?:\"[^\"]*\")|(?:'[^']*')|[^,]+)")


def _classify_default(param_name: str) -> tuple[str, str] | None:
    """Return (decision, rationale) for a defaulted parameter, or None if
    the parameter is out of ADR-0028's scope (keeps its default silently)."""
    if QUADRATURE_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: numerical-precision (quadrature node count) "
            "control; no in-repo citation for this value and/or below the "
            "AGENTS.md #1929 floor of 121 -> required argument.",
        )
    if CONVERGENCE_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: iteration/convergence precision control; no "
            "documented convergence-criterion source -> required argument.",
        )
    if REPLICATES_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 1: replicate/bootstrap/draw count; no "
            "Monte-Carlo-error source cited -> required argument.",
        )
    if SEED_RE.search(param_name):
        return (
            "require",
            "ADR-0028 rule 3: stochastic routines must not ship a default "
            "seed -> required argument.",
        )
    if MODEL_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 4: function is model-generic (not model-specific "
            "by name) -> model choice has no default.",
        )
    if THRESHOLD_RE.match(param_name):
        return (
            "require",
            "ADR-0028 rule 2: decision threshold / flag cutoff / stopping "
            "rule; no APA 7th source cited for this specific value -> "
            "required argument.",
        )
    return None


def _parse_params(parameters: str) -> list[tuple[str, str | None]]:
    """Return [(name, default_repr_or_None), ...] from an inventory row's
    ``parameters`` column."""
    out: list[tuple[str, str | None]] = []
    depth = 0
    current = ""
    parts = []
    for ch in parameters:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    for p in parts:
        if not p or p.startswith("*"):
            continue
        if "=" in p:
            name, default = p.split("=", 1)
            out.append((name.strip(), default.strip()))
        else:
            out.append((p.strip(), None))
    return out


def build_rows(inventory_rows: list[dict]) -> list[dict]:
    out_rows: list[dict] = []
    for r in inventory_rows:
        current_name = r["current_name"]
        module = r["module"]
        source = r["source"]
        kind = r["kind"]
        if source == "python" and current_name in RENAME_MAP:
            proposed_name = RENAME_MAP[current_name]
            name_decision = "rename"
            name_rationale = NAME_RATIONALE
        elif source == "pyo3":
            proposed_name = current_name
            name_decision = "keep"
            name_rationale = (
                "ADR-0028: PyO3 entry points are a private implementation "
                "detail, exempt from the public naming rules."
            )
        else:
            proposed_name = "keep"
            name_decision = "keep"
            name_rationale = "Already conforms to ADR-0028 naming convention."

        params = _parse_params(r.get("parameters", ""))
        defaulted = [(n, d) for n, d in params if d is not None]

        if not defaulted:
            out_rows.append(
                {
                    "source": source,
                    "current_name": current_name,
                    "proposed_name": proposed_name,
                    "name_decision": name_decision,
                    "name_rationale": name_rationale,
                    "module": module,
                    "kind": kind,
                    "parameter": "",
                    "current_default": "",
                    "default_decision": "n/a",
                    "default_rationale": "No defaulted parameters.",
                    "migration_note": "",
                }
            )
            continue

        for pname, pdefault in defaulted:
            classification = _classify_default(pname)
            if classification is None:
                decision, rationale = (
                    "keep",
                    "Out of ADR-0028 policy scope (not a precision control, "
                    "decision threshold, seed, or model-choice parameter).",
                )
            else:
                decision, rationale = classification
            migration_note = (
                (
                    f"Deprecation: keep {pname}={pdefault} accepted only when "
                    "explicitly passed for one minor release "
                    "(DeprecationWarning), then remove; changelog Changed "
                    "entry."
                )
                if decision in ("require", "change")
                else ""
            )
            out_rows.append(
                {
                    "source": source,
                    "current_name": current_name,
                    "proposed_name": proposed_name,
                    "name_decision": name_decision,
                    "name_rationale": name_rationale,
                    "module": module,
                    "kind": kind,
                    "parameter": pname,
                    "current_default": pdefault,
                    "default_decision": decision,
                    "default_rationale": rationale,
                    "migration_note": migration_note,
                }
            )
    out_rows.sort(key=lambda r: (r["module"], r["current_name"], r["parameter"]))
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    in_path = REPO_ROOT / "docs" / "api" / f"inventory-{args.date}.csv"
    out_path = REPO_ROOT / "docs" / "api" / f"renames-and-defaults-{args.date}.csv"

    with in_path.open(newline="") as fh:
        inventory_rows = list(csv.DictReader(fh))

    out_rows = build_rows(inventory_rows)

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
