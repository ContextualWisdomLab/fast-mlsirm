from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import FitConfig, MLS2PLMConfig
from .diagnostics import (dimensionality_diagnostics, fit_diagnostics,
                          response_process_dimensionality_diagnostics,
                          response_process_fit_diagnostics)
from .fit import fit
from .io import (load_factor_csv, load_params, save_dimensionality_diagnostics,
                 save_fit_diagnostics, save_fit_result, save_simulation)
from .report import render_diagnostics_report
from .simulation import simulate


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one machine-readable JSON status object to stdout.",
    )


def _progress(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "json", False):
        print(message)


def _complete(
    args: argparse.Namespace, message: str, payload: dict[str, object]
) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True))
    else:
        print(message)
    return 0


def _output_file(run_dir: str, filename: str) -> str:
    return str(Path(run_dir) / filename)


def _load_response_and_factors(
    responses_path: str, factors_path: str
) -> tuple[np.ndarray, np.ndarray]:
    responses = np.load(responses_path, allow_pickle=False)
    factors = load_factor_csv(factors_path)
    _validate_response_and_factors(responses, factors)
    return responses, factors


def _validate_response_and_factors(responses: np.ndarray, factors: np.ndarray) -> None:
    if responses.ndim != 2:
        raise ValueError("responses must be a 2D persons x items array")
    if factors.ndim != 1:
        raise ValueError("factor_id must be a 1D item vector")
    if factors.shape[0] != responses.shape[1]:
        raise ValueError(
            f"factor_id length ({factors.shape[0]}) must match response item count ({responses.shape[1]})"
        )


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
    sim.add_argument(
        "--persons",
        type=int,
        default=500,
        help="Number of persons to simulate (default: 500).",
    )
    sim.add_argument(
        "--dims",
        type=int,
        default=2,
        help="Number of true item dimensions (default: 2).",
    )
    sim.add_argument(
        "--items-per-dim",
        type=int,
        default=8,
        help="Number of items per dimension (default: 8).",
    )
    sim.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="Latent dimensionality for person traits (default: 2).",
    )
    sim.add_argument(
        "--phi",
        type=float,
        default=0.3,
        help="Variance of item intercept factors (default: 0.3).",
    )
    sim.add_argument(
        "--gamma",
        type=float,
        default=1.5,
        help="Variance of person trait coordinates (default: 1.5).",
    )
    sim.add_argument(
        "--seed", type=int, default=1, help="Random seed for simulation (default: 1)."
    )
    sim.add_argument(
        "--out",
        required=True,
        help="Directory path to save simulated output (responses, factors, etc.).",
    )
    _add_json_flag(sim)

    fit_cmd = sub.add_parser(
        "fit",
        help="Fit an MLSIRM model to response data.",
        description="Fit an MLSIRM model to response data.",
    )
    fit_cmd.add_argument(
        "--responses",
        required=True,
        help="Path to the responses numpy array file (.npy).",
    )
    fit_cmd.add_argument(
        "--factors", required=True, help="Path to the item factors CSV file."
    )
    fit_cmd.add_argument(
        "--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM)."
    )
    fit_cmd.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        help="Latent dimensionality for person traits (default: 2).",
    )
    fit_cmd.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs", "adam_lbfgs"],
        default="adam_lbfgs",
        help="Optimizer to use (default: adam_lbfgs).",
    )
    fit_cmd.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of iterations for the optimizer (default: 100).",
    )
    fit_cmd.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help="Number of random restarts (default: 1).",
    )
    fit_cmd.add_argument(
        "--seed", type=int, default=1, help="Random seed for fitting (default: 1)."
    )
    fit_cmd.add_argument(
        "--out", required=True, help="Directory path to save the fitted parameters."
    )
    _add_json_flag(fit_cmd)

    diagnose = sub.add_parser(
        "diagnose-fit",
        help="Compute item, person, and model fit diagnostics for fitted parameters.",
        description="Compute item, person, and model fit diagnostics for fitted parameters.",
    )
    diagnose.add_argument(
        "--responses",
        required=True,
        help="Path to the responses numpy array file (.npy).",
    )
    diagnose.add_argument(
        "--factors", required=True, help="Path to the item factors CSV file."
    )
    diagnose.add_argument("--params", required=True, help="Path to fitted params.npz.")
    diagnose.add_argument(
        "--model",
        default="MLS2PLM",
        help="Model type used for the fitted parameters (default: MLS2PLM).",
    )
    diagnose.add_argument(
        "--group-id", help="Optional .npy person group IDs for multigroup summaries."
    )
    diagnose.add_argument(
        "--cluster-id",
        help="Optional .npy person cluster IDs for multilevel summaries.",
    )
    diagnose.add_argument(
        "--out", required=True, help="Directory path to save fit_diagnostics.json."
    )
    _add_json_flag(diagnose)

    dim = sub.add_parser(
        "diagnose-dimensions",
        help="Compare latent-space dimensionality with K-fold held-out likelihood.",
        description="Compare latent-space dimensionality with K-fold held-out likelihood.",
    )
    dim.add_argument(
        "--responses",
        required=True,
        help="Path to the responses numpy array file (.npy).",
    )
    dim.add_argument(
        "--factors", required=True, help="Path to the item factors CSV file."
    )
    dim.add_argument(
        "--latent-dims",
        default="1,2,3",
        help="Comma-separated latent dimensions to compare (default: 1,2,3).",
    )
    dim.add_argument(
        "--folds", type=int, default=5, help="Number of validation folds (default: 5)."
    )
    dim.add_argument(
        "--model", default="MLS2PLM", help="Model type to fit (default: MLS2PLM)."
    )
    dim.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs", "adam_lbfgs"],
        default="adam_lbfgs",
        help="Optimizer to use (default: adam_lbfgs).",
    )
    dim.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations per fold fit (default: 100).",
    )
    dim.add_argument(
        "--n-restarts",
        type=int,
        default=1,
        help="Random restarts per fold fit (default: 1).",
    )
    dim.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for folds and fitting (default: 1).",
    )
    dim.add_argument(
        "--out",
        required=True,
        help="Directory path to save dimension_diagnostics.json.",
    )
    _add_json_flag(dim)

    process = sub.add_parser(
        "diagnose-response-process",
        help="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
        description="Compute dichotomous or polytomous fit diagnostics from category probabilities.",
    )
    process.add_argument(
        "--responses",
        required=True,
        help="Path to the responses numpy array file (.npy).",
    )
    process.add_argument(
        "--probabilities",
        required=True,
        help="Path to probabilities .npy, either persons x items or persons x items x categories.",
    )
    process.add_argument(
        "--item-type",
        choices=["dichotomous", "polytomous"],
        default="polytomous",
        help="Item type for metadata validation.",
    )
    process.add_argument(
        "--response-process",
        choices=["ideal_point", "cumulative"],
        default="cumulative",
        help="Response process represented by the probabilities.",
    )
    process.add_argument(
        "--group-id", help="Optional .npy person group IDs for multigroup summaries."
    )
    process.add_argument(
        "--cluster-id",
        help="Optional .npy person cluster IDs for multilevel summaries.",
    )
    process.add_argument(
        "--out", required=True, help="Directory path to save fit_diagnostics.json."
    )
    _add_json_flag(process)

    candidates = sub.add_parser(
        "diagnose-response-candidates",
        help="Compare response-process probability candidates with held-out likelihood.",
        description="Compare candidate category probability tensors for dimensionality or response-process checks.",
    )
    candidates.add_argument(
        "--responses",
        required=True,
        help="Path to the responses numpy array file (.npy).",
    )
    candidates.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate probability file as label=path.npy. Repeat for each candidate.",
    )
    candidates.add_argument(
        "--item-type",
        choices=["dichotomous", "polytomous"],
        default="polytomous",
        help="Item type for metadata validation.",
    )
    candidates.add_argument(
        "--response-process",
        choices=["ideal_point", "cumulative"],
        default="cumulative",
        help="Response process represented by the candidates.",
    )
    candidates.add_argument(
        "--out",
        required=True,
        help="Directory path to save dimension_diagnostics.json.",
    )
    _add_json_flag(candidates)

    report = sub.add_parser(
        "render-report",
        help="Render saved diagnostics JSON as a standalone HTML report.",
        description="Render fit_diagnostics.json or dimension_diagnostics.json as a standalone HTML report.",
    )
    report.add_argument(
        "--diagnostics",
        required=True,
        help="Path to fit_diagnostics.json or dimension_diagnostics.json.",
    )
    report.add_argument(
        "--out", required=True, help="Path to write the diagnostics HTML report."
    )
    report.add_argument("--title", help="Optional report title.")
    _add_json_flag(report)

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 2

    args = parser.parse_args(argv)
    if args.command == "simulate":
        _progress(
            args, f"⏳ Simulating {args.persons} persons and {args.dims} dimensions..."
        )
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
        return _complete(
            args,
            f"✅ Simulation successfully saved to {args.out}",
            {
                "command": "simulate",
                "status": "ok",
                "out": str(args.out),
                "n_persons": int(data.Y.shape[0]),
                "n_items": int(data.Y.shape[1]),
                "n_dims": int(data.config.n_dims),
                "files": {
                    "responses": _output_file(args.out, "responses.npy"),
                    "factors": _output_file(args.out, "item_factor.csv"),
                    "truth": _output_file(args.out, "truth.npz"),
                    "manifest": _output_file(args.out, "manifest.json"),
                },
            },
        )

    if args.command == "diagnose-fit":
        _progress(args, f"⏳ Computing {args.model} fit diagnostics...")
        try:
            responses, factors = _load_response_and_factors(
                args.responses, args.factors
            )
            params = load_params(args.params)
            group_id = _load_optional_npy(args.group_id)
            cluster_id = _load_optional_npy(args.cluster_id)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = fit_diagnostics(
            responses=responses,
            params=params,
            factor_id=factors,
            model=args.model,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        save_fit_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Fit diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-fit",
                "status": "ok",
                "out": str(args.out),
                "model": args.model,
                "files": {
                    "diagnostics": _output_file(args.out, "fit_diagnostics.json")
                },
            },
        )

    if args.command == "diagnose-dimensions":
        _progress(
            args, f"⏳ Comparing {args.model} latent dimensions {args.latent_dims}..."
        )
        try:
            responses, factors = _load_response_and_factors(
                args.responses, args.factors
            )
            latent_dims = [int(value) for value in args.latent_dims.split(",")]
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
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
        return _complete(
            args,
            f"✅ Dimension diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-dimensions",
                "status": "ok",
                "out": str(args.out),
                "model": args.model,
                "best_latent_dim": int(diagnostics.best["latent_dim"]),
                "files": {
                    "diagnostics": _output_file(args.out, "dimension_diagnostics.json")
                },
            },
        )

    if args.command == "diagnose-response-process":
        _progress(
            args,
            f"⏳ Computing {args.item_type} {args.response_process} fit diagnostics...",
        )
        try:
            responses = np.load(args.responses, allow_pickle=False)
            probabilities = np.load(args.probabilities, allow_pickle=False)
            group_id = _load_optional_npy(args.group_id)
            cluster_id = _load_optional_npy(args.cluster_id)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = response_process_fit_diagnostics(
            responses=responses,
            probabilities=probabilities,
            item_type=args.item_type,
            response_process=args.response_process,
            group_id=group_id,
            cluster_id=cluster_id,
        )
        save_fit_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Response process diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-response-process",
                "status": "ok",
                "out": str(args.out),
                "item_type": args.item_type,
                "response_process": args.response_process,
                "files": {
                    "diagnostics": _output_file(args.out, "fit_diagnostics.json")
                },
            },
        )

    if args.command == "diagnose-response-candidates":
        _progress(
            args,
            f"⏳ Comparing {args.item_type} {args.response_process} response candidates...",
        )
        try:
            responses = np.load(args.responses, allow_pickle=False)
            candidate_probabilities = _load_candidate_probabilities(args.candidate)
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to load data - {str(e)}", file=sys.stderr)
            return 1

        diagnostics = response_process_dimensionality_diagnostics(
            responses=responses,
            candidate_probabilities=candidate_probabilities,
            item_type=args.item_type,
            response_process=args.response_process,
        )
        save_dimensionality_diagnostics(diagnostics, args.out)
        return _complete(
            args,
            f"✅ Response candidate diagnostics successfully saved to {args.out}",
            {
                "command": "diagnose-response-candidates",
                "status": "ok",
                "out": str(args.out),
                "item_type": args.item_type,
                "response_process": args.response_process,
                "best_candidate": diagnostics.best["candidate_label"],
                "files": {
                    "diagnostics": _output_file(args.out, "dimension_diagnostics.json")
                },
            },
        )

    if args.command == "render-report":
        _progress(args, f"⏳ Rendering diagnostics report from {args.diagnostics}...")
        try:
            report_path = render_diagnostics_report(
                args.diagnostics, args.out, title=args.title
            )
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"❌ Error: Invalid diagnostics data - {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"❌ Error: Failed to render report - {str(e)}", file=sys.stderr)
            return 1

        return _complete(
            args,
            f"✅ Diagnostics report successfully saved to {report_path}",
            {
                "command": "render-report",
                "status": "ok",
                "out": str(report_path),
                "files": {
                    "report": str(report_path),
                    "diagnostics": str(args.diagnostics),
                },
            },
        )

    _progress(args, f"⏳ Fitting {args.model} model to responses...")
    try:
        responses, factors = _load_response_and_factors(args.responses, args.factors)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e.filename}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Error: Invalid input data - {str(e)}", file=sys.stderr)
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
    return _complete(
        args,
        f"✅ Fit result successfully saved to {args.out}",
        {
            "command": "fit",
            "status": "ok",
            "out": str(args.out),
            "model": result.model,
            "optimizer": result.optimizer,
            "objective": float(result.objective),
            "convergence_status": result.convergence_status,
            "n_iter": int(result.n_iter),
            "files": {
                "params": _output_file(args.out, "params.npz"),
                "summary": _output_file(args.out, "fit_summary.json"),
            },
        },
    )


def _load_optional_npy(path: str | None) -> np.ndarray | None:
    return None if path is None else np.load(path, allow_pickle=False)


def _load_candidate_probabilities(specs: list[str]) -> dict[str, np.ndarray]:
    candidates = {}
    for spec in specs:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        if not label:
            raise ValueError("candidate label must not be empty")
        if label in candidates:
            raise ValueError(f"duplicate candidate label: {label}")
        candidates[label] = np.load(path, allow_pickle=False)
    return candidates


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
