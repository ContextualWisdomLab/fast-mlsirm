from __future__ import annotations

import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .types import DimensionalityDiagnostics, FitDiagnostics, FitResult, MLSIRMParams, SimulationData


MAX_NUMPY_ARRAY_ELEMENTS = 50_000_000
MAX_NUMPY_ARRAY_BYTES = 512 * 1024 * 1024
MAX_NUMPY_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_NUMPY_ARCHIVE_MEMBERS = 256
MAX_NUMPY_HEADER_BYTES = 64 * 1024


def _validate_npy_header(stream: BinaryIO, source: str) -> tuple[int, int]:
    """Read only an NPY header and reject unsafe declared allocations."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=MAX_NUMPY_HEADER_BYTES
        )
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(
            stream, max_header_size=MAX_NUMPY_HEADER_BYTES
        )
    else:
        raise ValueError(f"{source} uses unsupported NPY format version {version}")
    if dtype.hasobject:
        raise ValueError(f"{source} contains an object dtype")

    elements = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"{source} declares a negative array dimension")
        if dim == 0:
            elements = 0
        elif elements and elements > MAX_NUMPY_ARRAY_ELEMENTS // dim:
            raise ValueError(
                f"{source} declares more than {MAX_NUMPY_ARRAY_ELEMENTS} array elements"
            )
        else:
            elements *= dim
    nbytes = elements * int(dtype.itemsize)
    if elements > MAX_NUMPY_ARRAY_ELEMENTS or nbytes > MAX_NUMPY_ARRAY_BYTES:
        raise ValueError(
            f"{source} declares {elements} elements / {nbytes} bytes, above the safe limit"
        )
    return nbytes, stream.tell()


def _validate_numpy_file(path: Path) -> None:
    file_size = path.stat().st_size
    if file_size > MAX_NUMPY_ARCHIVE_BYTES:
        raise ValueError(
            f"NumPy input exceeds the {MAX_NUMPY_ARCHIVE_BYTES}-byte file limit"
        )
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            nbytes, header_end = _validate_npy_header(stream, path.name)
        if file_size - header_end < nbytes:
            raise ValueError(
                f"{path.name} is truncated relative to its declared array shape"
            )
        return
    if path.suffix.lower() != ".npz":
        raise ValueError("NumPy input must use a .npy or .npz suffix")

    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members or len(members) > MAX_NUMPY_ARCHIVE_MEMBERS:
            raise ValueError(
                f"NPZ archive must contain 1..{MAX_NUMPY_ARCHIVE_MEMBERS} members"
            )
        total_bytes = 0
        for info in members:
            if not info.filename.endswith(".npy"):
                raise ValueError(
                    f"NPZ archive member {info.filename!r} is not an NPY array"
                )
            if info.file_size > MAX_NUMPY_ARRAY_BYTES + MAX_NUMPY_HEADER_BYTES:
                raise ValueError(
                    f"NPZ member {info.filename!r} exceeds the safe byte limit"
                )
            with archive.open(info) as stream:
                nbytes, header_end = _validate_npy_header(stream, info.filename)
            if info.file_size - header_end < nbytes:
                raise ValueError(
                    f"NPZ member {info.filename!r} is truncated relative to its declared array shape"
                )
            total_bytes += nbytes
            if total_bytes > MAX_NUMPY_ARCHIVE_BYTES:
                raise ValueError(
                    "NPZ archive declares more array bytes than the safe limit"
                )


def _load_numpy_bounded(path: str | Path):
    """Load NPY/NPZ only after validating headers and allocation bounds."""
    source = Path(path)
    _validate_numpy_file(source)
    return np.load(
        source,
        allow_pickle=False,
        max_header_size=MAX_NUMPY_HEADER_BYTES,
    )


def save_simulation(data: SimulationData, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(data.config), indent=2), encoding="utf-8")
    np.save(out / "responses.npy", data.Y)
    np.savez(
        out / "truth.npz",
        theta=data.truth.theta,
        alpha=data.truth.alpha,
        a=data.truth.a,
        b=data.truth.b,
        xi=data.truth.xi,
        zeta=data.truth.zeta,
        tau=np.array(data.truth.tau),
        gamma=np.array(data.truth.gamma),
        factor_id=data.factor_id,
        Phi=data.Phi,
    )
    _write_factor_csv(out / "item_factor.csv", data.factor_id)
    manifest = {
        "package": "fast-mlsirm",
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "MLS2PLM",
        "n_persons": int(data.Y.shape[0]),
        "n_items": int(data.Y.shape[1]),
        "n_dims": int(data.config.n_dims),
        "latent_dim": int(data.config.latent_dim),
        "gamma": float(data.config.gamma),
        "phi": float(data.config.phi),
        "seed": int(data.config.seed),
        "files": {"responses": "responses.npy", "truth": "truth.npz", "factors": "item_factor.csv"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_fit_result(result: FitResult, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = result.params
    arrays = dict(theta=p.theta, alpha=p.alpha, a=p.a, b=p.b, xi=p.xi, zeta=p.zeta, tau=p.tau, gamma=p.gamma)
    summary = {
        "model": result.model,
        "optimizer": result.optimizer,
        "backend": result.backend,
        "rust_device": result.rust_device,
        "objective": result.objective,
        "convergence_status": result.convergence_status,
        "n_iter": result.n_iter,
        "final_loglik": result.loglik_trace[-1] if result.loglik_trace else None,
    }
    if result.ic is not None:
        summary["information_criteria"] = {
            key: (float(v) if isinstance(v, float) else v)
            for key, v in result.ic.items()
        }
    if result.population is not None:
        pop = result.population
        summary["population"] = {"kind": pop["kind"]}
        for key in ("mu", "sigma", "u_eap", "theta_sd"):
            if key in pop:
                arrays[f"pop_{key}"] = np.asarray(pop[key])
        for key in ("sigma_u", "icc"):
            if key in pop:
                summary["population"][key] = float(pop[key])
    np.savez(out / "params.npz", **arrays)
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_fit_diagnostics(diagnostics: FitDiagnostics, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "itemfit": _arrays_to_lists(diagnostics.itemfit),
        "personfit": _arrays_to_lists(diagnostics.personfit),
        "factorfit": _arrays_to_lists(diagnostics.factorfit or {}),
        "categoryfit": _arrays_to_lists(diagnostics.categoryfit or {}),
        "groupfit": _arrays_to_lists(diagnostics.groupfit or {}),
        "clusterfit": _arrays_to_lists(diagnostics.clusterfit or {}),
        "group_itemfit": _arrays_to_lists(diagnostics.group_itemfit or {}),
        "cluster_itemfit": _arrays_to_lists(diagnostics.cluster_itemfit or {}),
        "model_fit": diagnostics.model_fit,
    }
    (out / "fit_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_dimensionality_diagnostics(diagnostics: DimensionalityDiagnostics, run_dir: str | Path) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": diagnostics.candidates, "best": diagnostics.best}
    (out / "dimension_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_params(path: str | Path) -> MLSIRMParams:
    with _load_numpy_bounded(path) as data:
        return MLSIRMParams(theta=data["theta"], alpha=data["alpha"], b=data["b"], xi=data["xi"], zeta=data["zeta"], tau=float(data["tau"]))


def load_factor_csv(path: str | Path) -> np.ndarray:
    import warnings
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError("factor CSV is empty")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.loadtxt(path, delimiter=',', skiprows=1, usecols=1, dtype=np.int64, ndmin=1)


def _write_factor_csv(path: Path, factor_id: np.ndarray) -> None:
    item_ids = np.arange(len(factor_id))
    data = np.column_stack((item_ids, factor_id))
    np.savetxt(
        path,
        data,
        delimiter=',',
        header='item_id,factor_id',
        comments='',
        fmt='%d'
    )


def _arrays_to_lists(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in values.items()}
