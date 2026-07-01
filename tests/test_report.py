import json

import pytest

from fast_mlsirm.report import render_diagnostics_report


def test_render_fit_diagnostics_report_has_sections(tmp_path):
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "itemfit": {"item_id": [0, 1], "outfit_mnsq": [1.0, 1.2], "observed_count": [4, 4]},
                "personfit": {"person_id": [0, 1], "outfit_mnsq": [0.9, 1.1], "observed_count": [2, 2]},
                "factorfit": {},
                "categoryfit": {},
                "groupfit": {},
                "clusterfit": {},
                "group_itemfit": {},
                "cluster_itemfit": {},
                "model_fit": {"loglik": -3.2, "deviance": 6.4},
            }
        ),
        encoding="utf-8",
    )

    rendered = render_diagnostics_report(source, out, title="Example Fit")

    assert rendered == out
    html = out.read_text(encoding="utf-8")
    assert "Example Fit" in html
    assert "Model Fit" in html
    assert "Item Fit" in html
    assert "Diagnostics Coverage" in html
    assert "No row data" in html
    assert "No rows were recorded in this section." not in html
    assert "No chartable values were recorded for this section." not in html
    assert "<table>" in html


def test_render_dimensionality_report_has_best_candidate(tmp_path):
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [
                    {"latent_dim": 1.0, "heldout_loglik": -12.5},
                    {"latent_dim": 2.0, "heldout_loglik": -8.0},
                ],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Dimensionality Diagnostics Report" in html
    assert "Best Candidate" in html
    assert "Candidate Comparison" in html
    assert "Latent Dim 2" in html
    assert "Latent Dim 2.0" not in html


def test_render_dimensionality_report_summarizes_empty_candidates(tmp_path):
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [],
                "best": {"latent_dim": 2.0, "heldout_loglik": -8.0},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Best Candidate" in html
    assert "Diagnostics Coverage" in html
    assert "Candidate Comparison" in html
    assert "<h2>Candidate Comparison</h2>" not in html
    assert "No rows were recorded in this section." not in html


def test_render_report_summarizes_empty_metric_sections(tmp_path):
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {},
                "itemfit": {"item_id": [0], "outfit_mnsq": [1.0], "observed_count": [4]},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Diagnostics Coverage" in html
    assert "No metric data" in html
    assert "Diagnostics without table rows or metric values are summarized here" in html
    no_metric_column = html[html.index("<h3>No metric data</h3>") :]
    no_metric_column = no_metric_column[: no_metric_column.index("</div>")]
    assert "Model Fit" in no_metric_column
    assert "<h2>Model Fit</h2>" not in html
    assert "No metrics were recorded in this diagnostics file." not in html


def test_render_dimensionality_report_summarizes_empty_best_candidate(tmp_path):
    source = tmp_path / "dimension_diagnostics.json"
    out = tmp_path / "dimensions.html"
    source.write_text(
        json.dumps(
            {
                "candidates": [{"latent_dim": 2.0, "heldout_loglik": -8.0}],
                "best": {},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "Diagnostics Coverage" in html
    assert "No metric data" in html
    no_metric_column = html[html.index("<h3>No metric data</h3>") :]
    no_metric_column = no_metric_column[: no_metric_column.index("</div>")]
    assert "Best Candidate" in no_metric_column
    assert "Candidate Comparison" in html
    assert "<h2>Best Candidate</h2>" not in html
    assert "<h2>Candidate Comparison</h2>" in html
    assert "No metrics were recorded in this diagnostics file." not in html


def test_render_table_section_omits_empty_chart_placeholder(tmp_path):
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.html"
    source.write_text(
        json.dumps(
            {
                "model_fit": {"loglik": -3.2},
                "itemfit": {"item_id": ["A"], "outfit_mnsq": [None], "observed_count": [4]},
            }
        ),
        encoding="utf-8",
    )

    render_diagnostics_report(source, out)

    html = out.read_text(encoding="utf-8")
    assert "<h2>Item Fit</h2>" in html
    assert "<table>" in html
    assert "No chartable values were recorded for this section." not in html


def test_render_report_rejects_unknown_payload(tmp_path):
    source = tmp_path / "unknown.json"
    out = tmp_path / "report.html"
    source.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported diagnostics JSON"):
        render_diagnostics_report(source, out)


def test_render_report_requires_html_output(tmp_path):
    source = tmp_path / "fit_diagnostics.json"
    out = tmp_path / "report.txt"
    source.write_text(json.dumps({"model_fit": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="must end with .html"):
        render_diagnostics_report(source, out)
