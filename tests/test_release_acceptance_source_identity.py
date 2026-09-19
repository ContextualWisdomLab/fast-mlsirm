from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import release_acceptance as subject


def test_release_acceptance_manifest_records_sealed_source_commit(
    monkeypatch, tmp_path: Path
) -> None:
    """Bind the standalone acceptance manifest to the exact source revision.

    The acquisition orchestrator may archive this manifest independently of its
    parent bundle. A consumer must therefore be able to reject evidence from a
    different checkout without relying on an outer manifest.
    """
    expected_commit = "a" * 40
    monkeypatch.setattr(subject, "_source_commit", lambda _repo_root: expected_commit)
    monkeypatch.setattr(
        subject,
        "read_json_object",
        lambda _path: {"backend": "rust"},
    )

    def fake_run_cli(
        arguments: list[str], out_label: str, *, require_json: bool = True
    ) -> dict[str, object]:
        del require_json
        output = Path(arguments[arguments.index("--out") + 1])
        if out_label == "simulate":
            output.mkdir(parents=True, exist_ok=True)
            (output / "responses.npy").write_bytes(b"responses")
            (output / "item_factor.csv").write_text("item,factor\n0,0\n", encoding="utf-8")
        elif out_label == "fit_auto":
            output.mkdir(parents=True, exist_ok=True)
            (output / "params.npz").write_bytes(b"params")
            (output / "fit_summary.json").write_text(
                json.dumps({"backend": "rust"}), encoding="utf-8"
            )
        elif out_label == "diagnose-fit":
            output.mkdir(parents=True, exist_ok=True)
            (output / "fit_diagnostics.json").write_text("{}", encoding="utf-8")
        elif out_label == "diagnose-dimensions":
            output.mkdir(parents=True, exist_ok=True)
            (output / "dimension_diagnostics.json").write_text("{}", encoding="utf-8")
        elif out_label in {"render-report-fit", "render-report-dimensions"}:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("<html></html>", encoding="utf-8")
        else:
            raise AssertionError(f"unexpected acceptance stage: {out_label}")
        payload: dict[str, object] = {"status": "ok"}
        if out_label == "fit_auto":
            payload["backend"] = "rust"
        return payload

    monkeypatch.setattr(subject, "_run_cli", fake_run_cli)
    out_dir = tmp_path / "acceptance"
    args = SimpleNamespace(
        out=str(out_dir),
        distribution_root=None,
        persons=2,
        dims=1,
        items_per_dim=2,
        latent_dim=1,
        seed=1,
        max_iter=1,
        n_restarts=1,
        latent_dims="1,2",
        folds=2,
        require_rust=False,
    )

    subject._run_acceptance(args)

    manifest = json.loads((out_dir / "acceptance_summary.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == expected_commit
