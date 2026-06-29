from __future__ import annotations

import argparse

import numpy as np

from .config import FitConfig, MLS2PLMConfig
from .fit import fit
from .io import load_factor_csv, save_fit_result, save_simulation
from .simulation import simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fast-mlsirm")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--persons", type=int, default=500)
    sim.add_argument("--dims", type=int, default=2)
    sim.add_argument("--items-per-dim", type=int, default=8)
    sim.add_argument("--latent-dim", type=int, default=2)
    sim.add_argument("--phi", type=float, default=0.3)
    sim.add_argument("--gamma", type=float, default=1.5)
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--out", required=True)

    fit_cmd = sub.add_parser("fit")
    fit_cmd.add_argument("--responses", required=True)
    fit_cmd.add_argument("--factors", required=True)
    fit_cmd.add_argument("--model", default="MLS2PLM")
    fit_cmd.add_argument("--latent-dim", type=int, default=2)
    fit_cmd.add_argument("--optimizer", choices=["adam", "lbfgs", "adam_lbfgs"], default="adam_lbfgs")
    fit_cmd.add_argument("--max-iter", type=int, default=100)
    fit_cmd.add_argument("--n-restarts", type=int, default=1)
    fit_cmd.add_argument("--seed", type=int, default=1)
    fit_cmd.add_argument("--out", required=True)

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
        return 0

    responses = np.load(args.responses)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
