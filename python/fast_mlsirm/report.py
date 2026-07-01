from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any


def render_diagnostics_report(
    diagnostics_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render saved diagnostics JSON as a standalone HTML report."""

    source = Path(diagnostics_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diagnostics JSON must contain an object")

    report_type = _diagnostics_type(payload)
    resolved_title = title or _default_title(report_type)

    out = Path(output_path)
    if out.suffix.lower() != ".html":
        raise ValueError("report output path must end with .html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _render_html(payload=payload, report_type=report_type, title=resolved_title, source_name=source.name),
        encoding="utf-8",
    )
    return out


def _diagnostics_type(payload: dict[str, Any]) -> str:
    if "model_fit" in payload:
        return "fit"
    if "candidates" in payload and "best" in payload:
        return "dimensions"
    raise ValueError("unsupported diagnostics JSON: expected fit or dimensionality diagnostics")


def _default_title(report_type: str) -> str:
    return "Fit Diagnostics Report" if report_type == "fit" else "Dimensionality Diagnostics Report"


def _render_html(payload: dict[str, Any], report_type: str, title: str, source_name: str) -> str:
    if report_type == "fit":
        sections = _render_fit_report(payload)
    else:
        sections = _render_dimensionality_report(payload)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(title)}</title>",
            "<style>",
            _css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            '<div class="hero-copy">',
            f"<p>fast-mlsirm diagnostics</p>",
            f"<h1>{escape(title)}</h1>",
            f"<span>Source: {escape(source_name)}</span>",
            "</div>",
            "</section>",
            *sections,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _render_fit_report(payload: dict[str, Any]) -> list[str]:
    model_fit = payload.get("model_fit", {})
    if not isinstance(model_fit, dict):
        raise ValueError("model_fit must be an object")

    sections = [
        _metric_section("Model Fit", model_fit),
        _table_section("Item Fit", _rows_from_columnar(payload.get("itemfit", {})), chart_value="outfit_mnsq"),
        _table_section("Person Fit", _rows_from_columnar(payload.get("personfit", {})), chart_value="outfit_mnsq"),
        _table_section("Factor Fit", _rows_from_columnar(payload.get("factorfit", {})), chart_value="outfit_mnsq"),
        _table_section("Category Fit", _rows_from_columnar(payload.get("categoryfit", {})), chart_value="outfit_mnsq"),
        _table_section("Group Fit", _rows_from_columnar(payload.get("groupfit", {})), chart_value="outfit_mnsq"),
        _table_section("Cluster Fit", _rows_from_columnar(payload.get("clusterfit", {})), chart_value="outfit_mnsq"),
        _table_section("Group Item Fit", _rows_from_columnar(payload.get("group_itemfit", {})), chart_value="outfit_mnsq"),
        _table_section(
            "Cluster Item Fit",
            _rows_from_columnar(payload.get("cluster_itemfit", {})),
            chart_value="outfit_mnsq",
        ),
    ]
    return sections


def _render_dimensionality_report(payload: dict[str, Any]) -> list[str]:
    best = payload.get("best", {})
    candidates = payload.get("candidates", [])
    if not isinstance(best, dict):
        raise ValueError("best must be an object")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")

    rows = [row for row in candidates if isinstance(row, dict)]
    return [
        _metric_section("Best Candidate", best),
        _table_section("Candidate Comparison", rows, chart_value="heldout_loglik"),
    ]


def _metric_section(heading: str, metrics: dict[str, Any]) -> str:
    cards = []
    for key, value in metrics.items():
        cards.append(
            "\n".join(
                [
                    '<article class="metric-card">',
                    f"<span>{escape(_label(key))}</span>",
                    f"<strong>{escape(_format_value(value))}</strong>",
                    "</article>",
                ]
            )
        )
    if not cards:
        cards.append('<p class="empty-state">No metrics were recorded in this diagnostics file.</p>')

    return "\n".join(
        [
            '<section class="report-section">',
            f"<h2>{escape(heading)}</h2>",
            '<div class="metrics-grid">',
            *cards,
            "</div>",
            "</section>",
        ]
    )


def _table_section(heading: str, rows: list[dict[str, Any]], *, chart_value: str | None = None) -> str:
    chart = _bar_chart(rows, chart_value) if chart_value else ""
    return "\n".join(
        [
            '<section class="report-section">',
            f"<h2>{escape(heading)}</h2>",
            chart,
            _table(rows),
            "</section>",
        ]
    )


def _bar_chart(rows: list[dict[str, Any]], value_key: str | None) -> str:
    if not value_key:
        return ""
    if not rows:
        return ""
    values = [float(row[value_key]) for row in rows if _is_number(row.get(value_key))]
    if not values:
        return '<p class="empty-state">No chartable values were recorded for this section.</p>'

    lower = min(values)
    upper = max(values)
    span = upper - lower
    chart_rows = []
    for index, row in enumerate(rows[:12]):
        raw_value = row.get(value_key)
        if not _is_number(raw_value):
            continue
        value = float(raw_value)
        width = 64.0 if span == 0 else 8.0 + ((value - lower) / span) * 92.0
        chart_rows.append(
            "\n".join(
                [
                    '<div class="bar-row">',
                    f'<span class="bar-label">{escape(_row_label(row, index))}</span>',
                    '<div class="bar-track">',
                    f'<div class="bar-fill" style="width: {width:.1f}%"></div>',
                    "</div>",
                    f'<span class="bar-value">{escape(_format_value(value))}</span>',
                    "</div>",
                ]
            )
        )

    if not chart_rows:
        return '<p class="empty-state">No chartable values were recorded for this section.</p>'

    return "\n".join(
        [
            '<div class="bar-chart" role="img" aria-label="Compact diagnostics bar chart">',
            *chart_rows,
            "</div>",
        ]
    )


def _table(rows: list[dict[str, Any]], *, limit: int = 12) -> str:
    if not rows:
        return '<p class="empty-state">No rows were recorded in this section.</p>'

    columns = _columns(rows)
    body_rows = []
    for row in rows[:limit]:
        cells = "".join(f"<td>{escape(_format_value(row.get(column, '')))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")

    note = ""
    if len(rows) > limit:
        note = f'<p class="table-note">Showing {limit} of {len(rows)} rows.</p>'

    headers = "".join(f"<th scope=\"col\">{escape(_label(column))}</th>" for column in columns)
    return "\n".join(
        [
            '<div class="table-wrap">',
            "<table>",
            f"<thead><tr>{headers}</tr></thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
            note,
        ]
    )


def _rows_from_columnar(section: Any) -> list[dict[str, Any]]:
    if not isinstance(section, dict) or not section:
        return []

    columns = list(section)
    lengths = [_value_length(section[column]) for column in columns]
    row_count = max(lengths) if lengths else 0
    if row_count == 0:
        return []

    rows = []
    for index in range(row_count):
        row = {}
        for column in columns:
            row[column] = _index_value(section[column], index)
        rows.append(row)
    return rows


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    ordered = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _value_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 1


def _index_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        return value[index] if index < len(value) else ""
    return value if index == 0 else ""


def _row_label(row: dict[str, Any], index: int) -> str:
    for key in ("candidate_label", "latent_dim", "item_id", "person_id", "factor_id", "category_id", "group_id", "cluster_id"):
        if key in row:
            return f"{_label(key)} {_format_label_value(row[key])}"
    return f"Row {index + 1}"


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4g}"
    if value is None:
        return ""
    return str(value)


def _format_label_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _format_value(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f8f5;
  --panel: #ffffff;
  --text: #202124;
  --muted: #60656f;
  --line: #d9ded6;
  --teal: #0f766e;
  --coral: #b45309;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.5;
}

main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}

.hero {
  min-height: 172px;
  display: flex;
  align-items: end;
  border-bottom: 4px solid var(--teal);
  margin-bottom: 24px;
  background: var(--panel);
  border-radius: 8px;
  padding: 28px;
}

.hero p,
.hero span {
  margin: 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.hero h1 {
  margin: 8px 0 12px;
  font-size: 3.2rem;
  line-height: 1;
  letter-spacing: 0;
}

.report-section {
  margin-top: 18px;
  padding: 22px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

h2 {
  margin: 0 0 16px;
  font-size: 1.18rem;
  letter-spacing: 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.metric-card {
  min-height: 96px;
  padding: 14px;
  border-left: 4px solid var(--coral);
  border-radius: 8px;
  background: #fafafa;
}

.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 0.82rem;
}

.metric-card strong {
  display: block;
  margin-top: 8px;
  font-size: 1.45rem;
  overflow-wrap: anywhere;
}

.bar-chart {
  display: grid;
  gap: 8px;
  margin-bottom: 16px;
}

.bar-row {
  display: grid;
  grid-template-columns: minmax(104px, 180px) 1fr minmax(64px, auto);
  gap: 10px;
  align-items: center;
}

.bar-label,
.bar-value,
.table-note,
.empty-state {
  color: var(--muted);
  font-size: 0.86rem;
}

.bar-track {
  height: 12px;
  overflow: hidden;
  background: #eef1eb;
  border-radius: 999px;
}

.bar-fill {
  height: 100%;
  min-width: 8px;
  background: var(--teal);
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 560px;
}

th,
td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}

th {
  background: #f1f4ef;
  color: #2f3437;
  font-size: 0.8rem;
}

tr:last-child td {
  border-bottom: 0;
}

.empty-state {
  margin: 0;
  padding: 14px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: #fbfcfa;
}

@media (max-width: 720px) {
  main {
    padding: 18px 12px 32px;
  }

  .hero,
  .report-section {
    padding: 18px;
  }

  .hero h1 {
    font-size: 2.1rem;
  }

  .bar-row {
    grid-template-columns: 1fr;
    gap: 4px;
  }
}
""".strip()
