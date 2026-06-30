from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import FitConfig, MLS2PLMConfig
from .diagnostics import dimensionality_diagnostics, fit_diagnostics, response_process_fit_diagnostics
from .fit import fit
from .io import load_factor_csv, load_params, save_dimensionality_diagnostics, save_fit_diagnostics, save_fit_result, save_simulation
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

    diagnose = sub.add_parser(
        "diagnose-fit",
        help="Compute item, person, and model fit diagnostics for fitted parameters.",
        description="Compute item, person, and model fit diagnostics for fitted parameters.",
    )
    diagnose.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    diagnose.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    diagnose.add_argument("--params", required=True, help="Path to fitted params.npz.")
    diagnose.add_argument("--model", default="MLS2PLM", help="Model type used for the fitted parameters (default: MLS2PLM).")
    diagnose.add_argument("--out", required=True, help="Directory path to save fit_diagnostics.json.")

    dim = sub.add_parser(
        "diagnose-dimensions",
        help="Compare latent-space dimensionality with K-fold held-out likelihood.",
        description="Compare latent-space dimensionality with K-fold held-out likelihood.",
    )
    dim.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    dim.add_argument("--factors", required=True, help="Path to the item factors CSV file.")
    dim.add_argument("--latent-dims", default="1,2,3", help="Comma-separated latent dimensions to compare (default: 1,2,3).")
    dim.add_argument("--folds", type=int, default=5, help="Number of validation folds (default: 5).")
    dim.add_argument("--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM).")
    dim.add_argument("--optimizer", choices=["adam", "lbfgs", "adam_lbfgs"], default="adam_lbfgs", help="Optimizer to use (default: adam_lbfgs).")
    dim.add_argument("--max-iter", type=int, default=100, help="Maximum iterations per fold fit (default: 100).")
    dim.add_argument("--n-restarts", type=int, default=1, help="Random restarts per fold fit (default: 1).")
    dim.add_argument("--seed", type=int, default=1, help="Random seed for folds and fitting (default: 1).")
    dim.add_argument("--out", required=True, help="Directory path to save dimension_diagnostics.json.")

    process = sub.add_parser(
        "diagnose-response-process",
        help="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
        description="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
    )
    process.add_argument("--responses", required=True, help="Path to the responses numpy array file (.npy).")
    process.add_argument("--probabilities", required=True, help="Path to probabilities .npy, either persons x items or persons x items x categories.")
    process.add_argument("--item-type", choices=["dichotomous", "polytomous"], default="polytomous", help="Item type for metadata validation.")
    process.add_argument("--response-process", choices=["ideal_point", "cumulative"], default="cumulative", help="Response process represented by the probabilities.")
    process.add_argument("--out", required=True, help="Directory path to save fit_diagnostics.json.")

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 2

    args = parser.parse_args(argv)
    if args.command == "simulate":
        print(f"⏳ Simulating {args.persons} persons and {args.dims} dimensions...")
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

    if args.command == "diagnose-fit":
        print(f"⏳ Computing {args.model} fit diagnostics...")
        try:
            responses = np.load(args.responses, allow_pickle=False)
            factors = load_factor_csv(args.factors)
            params = load_params(args.params)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = fit_diagnostics(responses=responses, params=params, factor_id=factors, model=args.model)
        save_fit_diagnostics(diagnostics, args.out)
        print(f"✅ Fit diagnostics successfully saved to {args.out}")
        return 0

    if args.command == "diagnose-dimensions":
        print(f"⏳ Comparing {args.model} latent dimensions {args.latent_dims}...")
        try:
            responses = np.load(args.responses, allow_pickle=False)
            factors = load_factor_csv(args.factors)
            latent_dims = [int(value) for value in args.latent_dims.split(",")]
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = dimensionality_diagnostics(
            responses=responses,
            factor_id=factors,
            latent_dims=latent_dims,
            model=args.model,
            k_folds=args.folds,
            seed=args.seed,
            config=FitConfig(
                model=args.model,
                optimizer=args.optimizer,
                max_iter=args.max_iter,
                n_restarts=args.n_restarts,
                seed=args.seed,
            ),
        )
        save_dimensionality_diagnostics(diagnostics, args.out)
        print(f"✅ Dimension diagnostics successfully saved to {args.out}")
        return 0

    if args.command == "diagnose-response-process":
        print(f"⏳ Computing {args.item_type} {args.response_process} fit diagnostics...")
        try:
            responses = np.load(args.responses, allow_pickle=False)
            probabilities = np.load(args.probabilities, allow_pickle=False)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = response_process_fit_diagnostics(
            responses=responses,
            probabilities=probabilities,
            item_type=args.item_type,
            response_process=args.response_process,
        )
        save_fit_diagnostics(diagnostics, args.out)
        print(f"✅ Response process diagnostics successfully saved to {args.out}")
        return 0

    print(f"⏳ Fitting {args.model} model to responses...")
    try:
        responses = np.load(args.responses, allow_pickle=False)
        factors = load_factor_csv(args.factors)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
        return 1

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


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
