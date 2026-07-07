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
            f'<meta http-equiv="Content-Security-Policy" content="{escape(_content_security_policy(), quote=True)}">',
            f"<title>{escape(title)}</title>",
            "<style>",
            _css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            '<div class="hero-copy">',
            "<p>fast-mlsirm diagnostics</p>",
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

    table_specs = [
        ("Item Fit", "itemfit", "outfit_mnsq"),
        ("Person Fit", "personfit", "outfit_mnsq"),
        ("Factor Fit", "factorfit", "outfit_mnsq"),
        ("Category Fit", "categoryfit", "outfit_mnsq"),
        ("Group Fit", "groupfit", "outfit_mnsq"),
        ("Cluster Fit", "clusterfit", "outfit_mnsq"),
        ("Group Item Fit", "group_itemfit", "outfit_mnsq"),
        ("Cluster Item Fit", "cluster_itemfit", "outfit_mnsq"),
    ]

    table_sections = []
    available = []
    no_row_tables = []
    for heading, payload_key, chart_value in table_specs:
        rows = _rows_from_columnar(payload.get(payload_key, {}))
        if rows:
            available.append(heading)
            table_sections.append(_table_section(heading, rows, chart_value=chart_value))
        else:
            no_row_tables.append(heading)

    sections = []
    no_metric_sections = []
    model_section = _metric_section("Model Fit", model_fit)
    if model_section:
        sections.append(model_section)
    else:
        no_metric_sections.append("Model Fit")

    if no_row_tables or no_metric_sections:
        sections.append(
            _availability_section(
                available=available,
                no_row_tables=no_row_tables,
                no_metric_sections=no_metric_sections,
            )
        )
    sections.extend(table_sections)
    return sections


def _render_dimensionality_report(payload: dict[str, Any]) -> list[str]:
    best = payload.get("best", {})
    candidates = payload.get("candidates", [])
    if not isinstance(best, dict):
        raise ValueError("best must be an object")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")

    rows = [row for row in candidates if isinstance(row, dict)]
    sections = []
    no_metric_sections = []
    best_section = _metric_section("Best Candidate", best)
    if best_section:
        sections.append(best_section)
    else:
        no_metric_sections.append("Best Candidate")

    if rows:
        sections.append(_table_section("Candidate Comparison", rows, chart_value="heldout_loglik"))
    if not rows or no_metric_sections:
        sections.append(
            _availability_section(
                available=["Candidate Comparison"] if rows else [],
                no_row_tables=[] if rows else ["Candidate Comparison"],
                no_metric_sections=no_metric_sections,
            )
        )
    return sections


def _metric_section(heading: str, metrics: dict[str, Any]) -> str | None:
    cards = []
    for key, value in metrics.items():
        cards.append(
            "\n".join(
                [
                    '<div class="metric-card">',
                    f"<dt>{escape(_label(key))}</dt>",
                    f"<dd>{escape(_format_value(value))}</dd>",
                    "</div>",
                ]
            )
        )
    if not cards:
        return None

    return "\n".join(
        [
            '<section class="report-section">',
            f"<h2>{escape(heading)}</h2>",
            '<dl class="metrics-grid">',
            *cards,
            "</dl>",
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
            _table(rows, label=f"{heading} diagnostics table"),
            "</section>",
        ]
    )


def _availability_section(
    *,
    available: list[str],
    no_row_tables: list[str],
    no_metric_sections: list[str] | None = None,
) -> str:
    no_metric_sections = no_metric_sections or []
    columns = []
    if available:
        columns.append(_coverage_column("Rendered tables", available))
    if no_row_tables:
        columns.append(_coverage_column("No row data", no_row_tables, muted=True))
    if no_metric_sections:
        columns.append(_coverage_column("No metric data", no_metric_sections, muted=True))

    return "\n".join(
        [
            '<section class="report-section report-coverage">',
            "<h2>Diagnostics Coverage</h2>",
            '<div class="coverage-grid">',
            *columns,
            "</div>",
            '<p class="coverage-note">Diagnostics without table rows or metric values are summarized here so the report does not render blank visual sections.</p>',
            "</section>",
        ]
    )


def _coverage_column(heading: str, items: list[str], *, muted: bool = False) -> str:
    class_name = "coverage-list coverage-list-muted" if muted else "coverage-list"
    item_markup = "\n".join(f"<li>{escape(name)}</li>" for name in items)
    return "\n".join(
        [
            '<div class="coverage-column">',
            f"<h3>{escape(heading)}</h3>",
            f'<ul class="{class_name}">{item_markup}</ul>',
            "</div>",
        ]
    )


def _bar_chart(rows: list[dict[str, Any]], value_key: str | None) -> str:
    if not value_key:
        return ""
    if not rows:
        return ""
    values = [float(row[value_key]) for row in rows if _is_number(row.get(value_key))]
    if not values:
        return ""

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
                    '<div class="bar-track" aria-hidden="true">',
                    f'<div class="bar-fill" style="width: {width:.1f}%"></div>',
                    "</div>",
                    f'<span class="bar-value">{escape(_format_value(value))}</span>',
                    "</div>",
                ]
            )
        )

    if not chart_rows:
        return ""

    return "\n".join(
        [
            '<div class="bar-chart" role="img" aria-label="Compact diagnostics bar chart">',
            *chart_rows,
            "</div>",
        ]
    )


def _table(rows: list[dict[str, Any]], *, label: str, limit: int = 12) -> str:
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
            f'<div class="table-wrap" role="region" aria-label="{escape(label)}" tabindex="0">',
            "<table>",
            f"<caption>{escape(label)}</caption>",
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


def _content_security_policy() -> str:
    return "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


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

h3 {
  margin: 0;
  color: #2f3437;
  font-size: 0.9rem;
  letter-spacing: 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 0;
}

.metric-card {
  min-height: 96px;
  padding: 14px;
  border-left: 4px solid var(--coral);
  border-radius: 8px;
  background: #fafafa;
}

.metric-card dt {
  display: block;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: normal;
  margin: 0;
}

.metric-card dd {
  display: block;
  margin: 8px 0 0 0;
  font-size: 1.45rem;
  font-weight: bold;
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
.coverage-note,
.empty-state {
  color: var(--muted);
  font-size: 0.86rem;
}

.coverage-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
}

.coverage-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 0;
  padding: 0;
  list-style: none;
}

.coverage-list li {
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #f1f4ef;
  color: #2f3437;
  font-size: 0.82rem;
}

.coverage-list-muted li {
  background: #fbfcfa;
  color: var(--muted);
}

.coverage-note {
  margin: 14px 0 0;
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

.table-wrap:focus-visible {
  outline: 3px solid #0f766e;
  outline-offset: 3px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 560px;
}

caption {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
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
