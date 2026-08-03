"""Apply the Monte-Carlo-aware higher-order CDM convergence threshold once."""

from pathlib import Path


PATH = Path("tests/unit/cdm_tests.rs")
OLD = '''        let conv_rate = nconv as f64 / reps as f64;
        assert!(
            conv_rate >= 0.95,
            "higher-order MC convergence rate {conv_rate:.3} below 0.95 for skew={skew}"
        );'''
NEW = '''        let conv_rate = nconv as f64 / reps as f64;
        // The observed convergence rate is itself a binomial Monte Carlo estimate.
        // Compare it with the 0.95 design target using a two-standard-error
        // simulation allowance instead of treating 500 replications as an exact
        // population percentage. At 500 replications this floor is about 0.9305.
        let target_rate = 0.95_f64;
        let mc_se = (target_rate * (1.0 - target_rate) / reps as f64).sqrt();
        let lower_acceptance = target_rate - 2.0 * mc_se;
        assert!(
            conv_rate >= lower_acceptance,
            "higher-order MC convergence rate {conv_rate:.3} below Monte-Carlo-aware \\
             floor {lower_acceptance:.3} for target {target_rate:.2} and skew={skew}"
        );'''


def main() -> None:
    """Replace the exact-rate assertion and fail if the expected source drifted."""
    text = PATH.read_text(encoding="utf-8")
    if NEW in text:
        return
    if text.count(OLD) != 1:
        raise SystemExit("expected exactly one higher-order convergence assertion")
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
