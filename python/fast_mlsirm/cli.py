from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import FitConfig, MLS2PLMConfig
from .fit import fit
from .io import load_factor_csv, save_fit_result, save_simulation
from .simulation import simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fast-mlsirm",
        description="Fast simulation, fitting, and recovery diagnostics for MLSIRM/MLS2PLM models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulate",
        help="Simulate binary responses for the MLS2PLM model.",
        description="Simulate binary responses for the MLS2PLM model.",
    )
    sim.add_argument("--persons", type=int, default=500, help="Number of persons to simulate (default: 500).")
    sim.add_argument("--dims", type=int, default=2, help="Number of true item dimensions (default: 2).")
    sim.add_argument("--items-per-dim", type=int, default=8, help="Number of items per dimension (default: 8).")
    sim.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality for person traits (default: 2).")
    sim.add_argument("--phi", type=float, default=0.3, help="Variance of item intercept factors (default: 0.3).")
    sim.add_argument("--gamma", type=float, default=1.5, help="Variance of person trait coordinates (default: 1.5).")
    sim.add_argument("--seed", type=int, default=1, help="Random seed for simulation (default: 1).")
    sim.add_argument("--out", required=True, help="Directory path to save simulated output (responses, factors, etc.).")

    fit_cmd = sub.add_parser(
        "fit",
        help="Fit an MLSIRM model to response data.",
        description="Fit an MLSIRM model to response data.",
    )
    fit_cmd.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    fit_cmd.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    fit_cmd.add_argument("--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM).")
    fit_cmd.add_argument("--latent-dim", type=int, default=2, help="Latent dimensionality for person traits (default: 2).")
    fit_cmd.add_argument("--optimizer", choices=["adam", "lbfgs", "adam_lbfgs"], default="adam_lbfgs", help="Optimizer to use (default: adam_lbfgs).")
    fit_cmd.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations for the optimizer (default: 100).")
    fit_cmd.add_argument("--n-restarts", type=int, default=1, help="Number of random restarts (default: 1).")
    fit_cmd.add_argument("--seed", type=int, default=1, help="Random seed for fitting (default: 1).")
    fit_cmd.add_argument("--out", required=True, help="Directory path to save the fitted parameters.")

    args = parser.parse_args(argv)
    if args.command == "simulate":
        data = simulate(
            MLS2PLMConfig(
                n_persons=args.persons,
                n_dims=args.dims,
                items_per_dim=args.items_per_dim,
                latent_dim=args.latent_dim,
                phi=args.phi,
                gamma=args.gamma,
                seed=args.seed,
            )
        )
        save_simulation(data, args.out)
        print(f"✅ Simulation successfully saved to {args.out}")
        return 0

    if not Path(args.responses).exists():
        print(f"❌ Error: Responses file '{args.responses}' not found.", file=sys.stderr)
        return 1
    if not Path(args.factors).exists():
        print(f"❌ Error: Factors file '{args.factors}' not found.", file=sys.stderr)
        return 1

    # Security: explicitly disable pickle to prevent arbitrary code execution
    responses = np.load(args.responses, allow_pickle=False)
    factors = load_factor_csv(args.factors)
    result = fit(
        responses=responses,
        factor_id=factors,
        config=FitConfig(
            model=args.model,
            latent_dim=args.latent_dim,
            optimizer=args.optimizer,
            max_iter=args.max_iter,
            n_restarts=args.n_restarts,
            seed=args.seed,
        ),
    )
    save_fit_result(result, args.out)
    print(f"✅ Fit result successfully saved to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
