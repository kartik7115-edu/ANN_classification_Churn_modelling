import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PARAMETERS_CSV_PATH = BASE_DIR / "logs" / "parameters.csv"
UPDATE_HISTORY_CSV_PATH = BASE_DIR / "logs" / "update_history.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "learning_diagnostics"

FEATURE_ORDER = ["Balance", "NumOfProducts", "HasCrCard", "IsActiveMember"]
OFFER_LABELS = {
    "cashback_bonus": "Cashback bonus",
    "fee_waiver": "Fee waiver",
    "rate_bonus": "Rate bonus",
    "credit_limit_review": "Credit limit review",
    "product_bundle": "Product bundle",
    "advisor_callback": "Advisor callback",
}
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#4338ca",
    "#0f766e",
]


def detect_python_command() -> str:
    if (BASE_DIR / ".venv312" / "Scripts" / "python.exe").exists():
        return r".\.venv312\Scripts\python.exe"
    if (BASE_DIR / ".venv" / "Scripts" / "python.exe").exists():
        return r".\.venv\Scripts\python.exe"
    return "python"


def parse_json_cell(value: Any) -> dict[str, Any]:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(value)


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def padded_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.15, 0.1)
    else:
        padding = (maximum - minimum) * 0.12
    return minimum - padding, maximum + padding


def make_ticks(minimum: float, maximum: float, count: int = 5) -> list[float]:
    if count <= 1 or minimum == maximum:
        return [minimum]
    step = (maximum - minimum) / (count - 1)
    return [minimum + step * idx for idx in range(count)]


def line_chart_svg(
    rows: list[dict[str, Any]],
    series_key: str,
    x_key: str,
    y_key: str,
    title: str,
    subtitle: str,
    y_label: str,
    output_path: Path,
    x_label: str = "Batch number",
    series_order: list[str] | None = None,
    width: int = 1200,
    height: int = 720,
) -> None:
    if not rows:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    left = 92
    right = 300
    top = 96
    bottom = 88
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = [float(row[y_key]) for row in rows]
    min_y, max_y = padded_range(y_values)
    min_x, max_x = min(x_values), max(x_values)
    if series_order:
        seen = {str(row[series_key]) for row in rows}
        series_names = [name for name in series_order if name in seen]
        series_names.extend(sorted(seen - set(series_names)))
    else:
        series_names = sorted({str(row[series_key]) for row in rows})
    color_map = {name: COLORS[idx % len(COLORS)] for idx, name in enumerate(series_names)}

    def x_pos(value: float) -> float:
        if min_x == max_x:
            return left + plot_width / 2
        return left + ((value - min_x) / (max_x - min_x)) * plot_width

    def y_pos(value: float) -> float:
        return top + (1 - ((value - min_y) / (max_y - min_y))) * plot_height

    elements: list[str] = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f"<text x=\"{left}\" y=\"42\" font-family=\"Segoe UI, Arial\" font-size=\"28\" font-weight=\"700\" fill=\"#111827\">{escape(title)}</text>",
        f"<text x=\"{left}\" y=\"70\" font-family=\"Segoe UI, Arial\" font-size=\"14\" fill=\"#4b5563\">{escape(subtitle)}</text>",
    ]

    for tick in make_ticks(min_y, max_y):
        y = y_pos(tick)
        elements.append(f"<line x1=\"{left}\" y1=\"{y:.2f}\" x2=\"{left + plot_width}\" y2=\"{y:.2f}\" stroke=\"#e5e7eb\"/>")
        elements.append(f"<text x=\"{left - 12}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-family=\"Segoe UI, Arial\" font-size=\"12\" fill=\"#4b5563\">{fmt(tick)}</text>")

    for batch in x_values:
        x = x_pos(batch)
        elements.append(f"<line x1=\"{x:.2f}\" y1=\"{top}\" x2=\"{x:.2f}\" y2=\"{top + plot_height}\" stroke=\"#f3f4f6\"/>")
        elements.append(f"<text x=\"{x:.2f}\" y=\"{top + plot_height + 28}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial\" font-size=\"12\" fill=\"#4b5563\">{int(batch)}</text>")

    elements.extend(
        [
            f"<line x1=\"{left}\" y1=\"{top + plot_height}\" x2=\"{left + plot_width}\" y2=\"{top + plot_height}\" stroke=\"#111827\"/>",
            f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_height}\" stroke=\"#111827\"/>",
            f"<text x=\"{left + plot_width / 2}\" y=\"{height - 24}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial\" font-size=\"14\" fill=\"#111827\">{escape(x_label)}</text>",
            f"<text transform=\"translate(24 {top + plot_height / 2}) rotate(-90)\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial\" font-size=\"14\" fill=\"#111827\">{escape(y_label)}</text>",
        ]
    )

    for name in series_names:
        points = sorted(
            [(float(row[x_key]), float(row[y_key])) for row in rows if str(row[series_key]) == name],
            key=lambda pair: pair[0],
        )
        point_string = " ".join(f"{x_pos(x):.2f},{y_pos(y):.2f}" for x, y in points)
        color = color_map[name]
        elements.append(f"<polyline points=\"{point_string}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>")
        for x, y in points:
            elements.append(f"<circle cx=\"{x_pos(x):.2f}\" cy=\"{y_pos(y):.2f}\" r=\"4\" fill=\"{color}\" stroke=\"#ffffff\" stroke-width=\"1.5\"/>")

    legend_x = left + plot_width + 36
    legend_y = top + 10
    elements.append(f"<text x=\"{legend_x}\" y=\"{legend_y}\" font-family=\"Segoe UI, Arial\" font-size=\"14\" font-weight=\"700\" fill=\"#111827\">Legend</text>")
    for idx, name in enumerate(series_names):
        y = legend_y + 28 + idx * 24
        color = color_map[name]
        elements.append(f"<line x1=\"{legend_x}\" y1=\"{y}\" x2=\"{legend_x + 24}\" y2=\"{y}\" stroke=\"{color}\" stroke-width=\"3\"/>")
        elements.append(f"<text x=\"{legend_x + 34}\" y=\"{y + 4}\" font-family=\"Segoe UI, Arial\" font-size=\"12\" fill=\"#374151\">{escape(name)}</text>")

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def offer_small_multiples_svg(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    features = [feature for feature in FEATURE_ORDER if any(row["feature"] == feature for row in rows)]
    offers = sorted({row["offer"] for row in rows})
    color_map = {offer: COLORS[idx % len(COLORS)] for idx, offer in enumerate(offers)}
    x_values = sorted({float(row["batch_number"]) for row in rows})
    y_values = [float(row["weight"]) for row in rows]
    min_y, max_y = padded_range(y_values)
    min_x, max_x = min(x_values), max(x_values)

    width = 1300
    height = 920
    margin_x = 74
    top = 136
    panel_gap_x = 58
    panel_gap_y = 72
    panel_width = (width - margin_x * 2 - panel_gap_x) / 2
    panel_height = 255

    def x_pos(value: float, panel_left: float) -> float:
        if min_x == max_x:
            return panel_left + panel_width / 2
        return panel_left + ((value - min_x) / (max_x - min_x)) * panel_width

    def y_pos(value: float, panel_top: float) -> float:
        return panel_top + (1 - ((value - min_y) / (max_y - min_y))) * panel_height

    elements: list[str] = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        "<text x=\"74\" y=\"42\" font-family=\"Segoe UI, Arial\" font-size=\"28\" font-weight=\"700\" fill=\"#111827\">Offer Parameter Updates</text>",
        "<text x=\"74\" y=\"70\" font-family=\"Segoe UI, Arial\" font-size=\"14\" fill=\"#4b5563\">Each panel tracks learned offer weights for one feature. X-axis is batch number.</text>",
    ]

    legend_x = 74
    legend_y = 104
    for idx, offer in enumerate(offers):
        x = legend_x + idx * 190
        label = OFFER_LABELS.get(offer, offer.replace("_", " ").title())
        elements.append(f"<line x1=\"{x}\" y1=\"{legend_y}\" x2=\"{x + 28}\" y2=\"{legend_y}\" stroke=\"{color_map[offer]}\" stroke-width=\"3\"/>")
        elements.append(f"<text x=\"{x + 36}\" y=\"{legend_y + 4}\" font-family=\"Segoe UI, Arial\" font-size=\"12\" fill=\"#374151\">{escape(label)}</text>")

    for feature_idx, feature in enumerate(features):
        col = feature_idx % 2
        row = feature_idx // 2
        panel_left = margin_x + col * (panel_width + panel_gap_x)
        panel_top = top + row * (panel_height + panel_gap_y)

        elements.append(f"<text x=\"{panel_left}\" y=\"{panel_top - 18}\" font-family=\"Segoe UI, Arial\" font-size=\"18\" font-weight=\"700\" fill=\"#111827\">{escape(feature)}</text>")
        elements.append(f"<rect x=\"{panel_left}\" y=\"{panel_top}\" width=\"{panel_width}\" height=\"{panel_height}\" fill=\"#ffffff\" stroke=\"#d1d5db\"/>")

        for tick in make_ticks(min_y, max_y, count=4):
            y = y_pos(tick, panel_top)
            elements.append(f"<line x1=\"{panel_left}\" y1=\"{y:.2f}\" x2=\"{panel_left + panel_width}\" y2=\"{y:.2f}\" stroke=\"#e5e7eb\"/>")
            elements.append(f"<text x=\"{panel_left - 10}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-family=\"Segoe UI, Arial\" font-size=\"11\" fill=\"#4b5563\">{fmt(tick)}</text>")

        for batch in x_values:
            x = x_pos(batch, panel_left)
            elements.append(f"<text x=\"{x:.2f}\" y=\"{panel_top + panel_height + 24}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial\" font-size=\"11\" fill=\"#4b5563\">{int(batch)}</text>")

        for offer in offers:
            points = sorted(
                [
                    (float(item["batch_number"]), float(item["weight"]))
                    for item in rows
                    if item["feature"] == feature and item["offer"] == offer
                ],
                key=lambda pair: pair[0],
            )
            if not points:
                continue
            point_string = " ".join(f"{x_pos(x, panel_left):.2f},{y_pos(y, panel_top):.2f}" for x, y in points)
            color = color_map[offer]
            elements.append(f"<polyline points=\"{point_string}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.4\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>")
            for x, y in points:
                elements.append(f"<circle cx=\"{x_pos(x, panel_left):.2f}\" cy=\"{y_pos(y, panel_top):.2f}\" r=\"3\" fill=\"{color}\"/>")

    elements.append(f"<text x=\"{width / 2}\" y=\"{height - 24}\" text-anchor=\"middle\" font-family=\"Segoe UI, Arial\" font-size=\"14\" fill=\"#111827\">Batch number</text>")
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def build_parameter_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    params_df = pd.read_csv(PARAMETERS_CSV_PATH)
    feature_rows: list[dict[str, Any]] = []
    offer_rows: list[dict[str, Any]] = []

    for batch_number, (_, row) in enumerate(params_df.iterrows(), start=1):
        feature_weights = parse_json_cell(row["feature_weights"])
        offer_weights = parse_json_cell(row["offer_weights"])

        for feature, weight in feature_weights.items():
            feature_rows.append(
                {
                    "batch_number": batch_number,
                    "feature": feature,
                    "weight": float(weight),
                    "reason": row.get("reason", ""),
                    "customer_id": row.get("customer_id", ""),
                }
            )

        for feature, offers in offer_weights.items():
            for offer, weight in offers.items():
                offer_rows.append(
                    {
                        "batch_number": batch_number,
                        "feature": feature,
                        "offer": offer,
                        "offer_label": OFFER_LABELS.get(offer, offer.replace("_", " ").title()),
                        "weight": float(weight),
                        "reason": row.get("reason", ""),
                        "customer_id": row.get("customer_id", ""),
                    }
                )

    return pd.DataFrame(feature_rows), pd.DataFrame(offer_rows)


def build_feedback_frame() -> pd.DataFrame:
    if not UPDATE_HISTORY_CSV_PATH.exists():
        return pd.DataFrame()

    updates_df = pd.read_csv(UPDATE_HISTORY_CSV_PATH)
    if updates_df.empty:
        return updates_df

    updates_df["interaction_number"] = range(1, len(updates_df) + 1)
    updates_df["accepted_offer"] = updates_df["accepted_offer"].astype(int)
    updates_df["pre_churn"] = updates_df["pre_churn"].astype(float)
    updates_df["current_churn"] = updates_df["current_churn"].astype(float)
    updates_df["churn_delta"] = updates_df["current_churn"] - updates_df["pre_churn"]
    updates_df["accepted_count"] = updates_df["accepted_offer"].cumsum()
    updates_df["rejected_count"] = updates_df["interaction_number"] - updates_df["accepted_count"]
    return updates_df


def write_report_readme(files: dict[str, Path]) -> None:
    python_command = detect_python_command()
    lines = [
        "# Learning Agent Diagnostics",
        "",
        "These charts are generated for the project report and internal analysis. They are intentionally separate from the Streamlit demo UI.",
        "",
        "## Generated Charts",
        "",
        f"- Feature parameter updates: [{files['feature_chart'].name}]({files['feature_chart'].name})",
        f"- Offer parameter updates: [{files['offer_chart'].name}]({files['offer_chart'].name})",
        f"- Before vs after churn risk: [{files['churn_chart'].name}]({files['churn_chart'].name})",
        f"- Accepted vs rejected feedback trend: [{files['feedback_chart'].name}]({files['feedback_chart'].name})",
        "",
        "## Data Exports",
        "",
        f"- Flattened feature weights: [{files['feature_csv'].name}]({files['feature_csv'].name})",
        f"- Flattened offer weights: [{files['offer_csv'].name}]({files['offer_csv'].name})",
        f"- Feedback summary: [{files['feedback_csv'].name}]({files['feedback_csv'].name})",
        "",
        "## Regenerate",
        "",
        "```powershell",
        f"{python_command} report_visuals.py",
        "```",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class=\"empty-state\">No rows available.</p>"
    return df.to_html(index=False, classes="data-table", border=0, justify="left", escape=True)


def write_index_html(
    feature_df: pd.DataFrame,
    offer_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    files: dict[str, Path],
) -> None:
    feature_summary = (
        f"{feature_df['batch_number'].nunique()} batches"
        if not feature_df.empty and "batch_number" in feature_df
        else "No feature updates"
    )
    offer_summary = (
        f"{offer_df['offer'].nunique()} offers across {offer_df['feature'].nunique()} features"
        if not offer_df.empty
        else "No offer updates"
    )
    feedback_summary = (
        f"{int(feedback_df['accepted_offer'].sum())} accepted / {len(feedback_df) - int(feedback_df['accepted_offer'].sum())} rejected"
        if not feedback_df.empty and "accepted_offer" in feedback_df
        else "No feedback history"
    )

    feature_preview = feature_df.round({"weight": 4}).copy()
    offer_preview = offer_df.round({"weight": 4}).copy()
    feedback_preview = feedback_df.round({"pre_churn": 4, "current_churn": 4, "churn_delta": 4}).copy()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Learning Diagnostics</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe6;
      --panel: #fffdf9;
      --ink: #1f2937;
      --muted: #5b6472;
      --line: #ddd3c5;
      --accent: #0f766e;
      --accent-soft: #d9f3ee;
      --shadow: 0 18px 40px rgba(31, 41, 55, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1400px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.96), rgba(17, 24, 39, 0.96));
      color: #f8fafc;
      border-radius: 24px;
      padding: 28px 32px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      max-width: 880px;
      color: rgba(248, 250, 252, 0.84);
      line-height: 1.6;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 22px 0 0;
    }}
    .stat {{
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 18px;
      padding: 16px 18px;
      backdrop-filter: blur(10px);
    }}
    .stat-label {{
      margin: 0 0 6px;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(248, 250, 252, 0.68);
    }}
    .stat-value {{
      margin: 0;
      font-size: 1.1rem;
      font-weight: 700;
    }}
    .section {{
      margin-top: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-top: 24px;
      background: rgba(255, 255, 255, 0.18);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 18px;
      padding: 14px;
    }}
    .toggle-group {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .toggle-btn {{
      appearance: none;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.08);
      color: #f8fafc;
      padding: 10px 14px;
      border-radius: 999px;
      font: inherit;
      cursor: pointer;
      font-weight: 600;
    }}
    .toggle-btn.active {{
      background: #f8fafc;
      color: #111827;
      border-color: #f8fafc;
    }}
    .toggle-note {{
      margin: 0;
      color: rgba(248, 250, 252, 0.75);
      font-size: 0.95rem;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 1.4rem;
    }}
    .section p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .chart-frame {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #fff;
    }}
    .chart-frame object {{
      display: block;
      width: 100%;
      min-height: 720px;
    }}
    .downloads {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .downloads a {{
      text-decoration: none;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid rgba(15, 118, 110, 0.18);
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 600;
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 16px;
      margin-top: 18px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--ink);
    }}
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
    }}
    .filter-input {{
      flex: 1 1 280px;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    .filter-count {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .table-wrap {{
      margin-top: 14px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fff;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    .data-table th,
    .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #ebe3d7;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      position: sticky;
      top: 0;
      background: #fbf8f2;
      color: #374151;
      z-index: 1;
    }}
    .data-table tr:nth-child(even) td {{
      background: #fffcf7;
    }}
    .empty-state {{
      color: var(--muted);
      margin: 10px 0 0;
    }}
    body.mode-charts .data-section,
    body.mode-charts details {{
      display: none;
    }}
    body.mode-data .chart-section {{
      display: none;
    }}
    .hidden-row {{
      display: none;
    }}
    @media (max-width: 900px) {{
      .page {{ width: min(100% - 20px, 1400px); padding-top: 20px; }}
      .hero, .section {{ border-radius: 18px; padding: 18px; }}
      .chart-frame object {{ min-height: 460px; }}
      .toolbar {{ padding: 12px; }}
    }}
  </style>
</head>
<body class="mode-all">
  <main class="page">
    <section class="hero">
      <h1>Learning Diagnostics Dashboard</h1>
      <p>Single-page view for retention-agent parameter trends, offer weight evolution, churn comparisons, and exported diagnostic tables.</p>
      <div class="stats">
        <article class="stat">
          <p class="stat-label">Feature Trends</p>
          <p class="stat-value">{escape(feature_summary)}</p>
        </article>
        <article class="stat">
          <p class="stat-label">Offer Trends</p>
          <p class="stat-value">{escape(offer_summary)}</p>
        </article>
        <article class="stat">
          <p class="stat-label">Feedback</p>
          <p class="stat-value">{escape(feedback_summary)}</p>
        </article>
      </div>
      <div class="toolbar">
        <div class="toggle-group" aria-label="Display mode">
          <button class="toggle-btn active" type="button" data-mode="all">Charts + Data</button>
          <button class="toggle-btn" type="button" data-mode="charts">Charts Only</button>
          <button class="toggle-btn" type="button" data-mode="data">Data Only</button>
        </div>
        <p class="toggle-note">Use view mode to focus on visuals or inspect the exported tables.</p>
      </div>
    </section>

    <section class="section">
      <h2>Feature Parameter Updates</h2>
      <p>Line chart showing how the learned feature weights move across batches.</p>
      <div class="chart-frame chart-section">
        <object type="image/svg+xml" data="{files['feature_chart'].name}">Feature parameter SVG</object>
      </div>
      <details class="data-section">
        <summary>Feature Parameter CSV Preview</summary>
        <div class="filter-bar">
          <input class="filter-input" type="search" data-table-target="feature-table" placeholder="Filter feature rows by batch, feature, reason, or customer id">
          <span class="filter-count" data-count-target="feature-table">Showing all rows</span>
        </div>
        <div class="table-wrap" id="feature-table">{build_html_table(feature_preview)}</div>
      </details>
    </section>

    <section class="section">
      <h2>Offer Parameter Updates</h2>
      <p>Small multiples view of offer weights by feature, with the exported flattened table below.</p>
      <div class="chart-frame chart-section">
        <object type="image/svg+xml" data="{files['offer_chart'].name}">Offer parameter SVG</object>
      </div>
      <details class="data-section">
        <summary>Offer Parameter CSV Preview</summary>
        <div class="filter-bar">
          <input class="filter-input" type="search" data-table-target="offer-table" placeholder="Filter offer rows by batch, feature, offer, weight, or reason">
          <span class="filter-count" data-count-target="offer-table">Showing all rows</span>
        </div>
        <div class="table-wrap" id="offer-table">{build_html_table(offer_preview)}</div>
      </details>
    </section>

    <section class="section">
      <h2>Feedback Diagnostics</h2>
      <p>Comparison of churn probability before and after offers, plus acceptance and rejection trend counts.</p>
      <div class="chart-frame chart-section">
        <object type="image/svg+xml" data="{files['churn_chart'].name}">Churn before/after SVG</object>
      </div>
      <div class="chart-frame chart-section" style="margin-top: 18px;">
        <object type="image/svg+xml" data="{files['feedback_chart'].name}">Feedback acceptance trend SVG</object>
      </div>
      <details class="data-section">
        <summary>Feedback Summary CSV Preview</summary>
        <div class="filter-bar">
          <input class="filter-input" type="search" data-table-target="feedback-table" placeholder="Filter feedback rows by customer, offer, acceptance, or churn values">
          <span class="filter-count" data-count-target="feedback-table">Showing all rows</span>
        </div>
        <div class="table-wrap" id="feedback-table">{build_html_table(feedback_preview)}</div>
      </details>
    </section>

    <section class="section">
      <h2>Downloads</h2>
      <p>Direct links to the generated report assets.</p>
      <div class="downloads">
        <a href="{files['feature_chart'].name}">Feature SVG</a>
        <a href="{files['offer_chart'].name}">Offer SVG</a>
        <a href="{files['churn_chart'].name}">Churn SVG</a>
        <a href="{files['feedback_chart'].name}">Feedback SVG</a>
        <a href="{files['feature_csv'].name}">Feature CSV</a>
        <a href="{files['offer_csv'].name}">Offer CSV</a>
        <a href="{files['feedback_csv'].name}">Feedback CSV</a>
      </div>
    </section>
  </main>
  <script>
    (function () {{
      const body = document.body;
      const buttons = Array.from(document.querySelectorAll(".toggle-btn"));
      const setMode = (mode) => {{
        body.classList.remove("mode-all", "mode-charts", "mode-data");
        body.classList.add(`mode-${{mode}}`);
        buttons.forEach((button) => {{
          button.classList.toggle("active", button.dataset.mode === mode);
        }});
      }};
      buttons.forEach((button) => {{
        button.addEventListener("click", () => setMode(button.dataset.mode || "all"));
      }});

      const updateFilter = (tableId, query) => {{
        const tableRoot = document.getElementById(tableId);
        if (!tableRoot) return;
        const table = tableRoot.querySelector("table");
        if (!table) return;
        const rows = Array.from(table.querySelectorAll("tbody tr"));
        const normalizedQuery = query.trim().toLowerCase();
        let visibleCount = 0;
        rows.forEach((row) => {{
          const text = row.textContent.toLowerCase();
          const visible = !normalizedQuery || text.includes(normalizedQuery);
          row.classList.toggle("hidden-row", !visible);
          if (visible) visibleCount += 1;
        }});
        const countNode = document.querySelector(`[data-count-target="${{tableId}}"]`);
        if (countNode) {{
          countNode.textContent = normalizedQuery
            ? `Showing ${{visibleCount}} of ${{rows.length}} rows`
            : `Showing all ${{rows.length}} rows`;
        }}
      }};

      document.querySelectorAll(".filter-input").forEach((input) => {{
        const tableId = input.getAttribute("data-table-target");
        if (!tableId) return;
        updateFilter(tableId, "");
        input.addEventListener("input", () => updateFilter(tableId, input.value));
      }});
    }})();
  </script>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df, offer_df = build_parameter_frames()
    feedback_df = build_feedback_frame()

    feature_csv = OUTPUT_DIR / "feature_parameter_updates.csv"
    offer_csv = OUTPUT_DIR / "offer_parameter_updates.csv"
    feedback_csv = OUTPUT_DIR / "feedback_summary.csv"

    feature_df.to_csv(feature_csv, index=False)
    offer_df.to_csv(offer_csv, index=False)
    feedback_df.to_csv(feedback_csv, index=False)

    feature_chart = OUTPUT_DIR / "feature_parameter_updates.svg"
    offer_chart = OUTPUT_DIR / "offer_parameter_updates.svg"
    churn_chart = OUTPUT_DIR / "churn_before_after.svg"
    feedback_chart = OUTPUT_DIR / "feedback_acceptance_trend.svg"

    line_chart_svg(
        rows=feature_df.to_dict("records"),
        series_key="feature",
        x_key="batch_number",
        y_key="weight",
        title="Feature Parameter Updates",
        subtitle="Feature weights learned by the retention agent. X-axis is batch number.",
        y_label="Feature weight",
        output_path=feature_chart,
        x_label="Batch number",
    )

    offer_small_multiples_svg(offer_df.to_dict("records"), offer_chart)

    if not feedback_df.empty:
        churn_rows = []
        for _, row in feedback_df.iterrows():
            churn_rows.append(
                {
                    "interaction_number": int(row["interaction_number"]),
                    "series": "Before offer",
                    "churn": float(row["pre_churn"]),
                }
            )
            churn_rows.append(
                {
                    "interaction_number": int(row["interaction_number"]),
                    "series": "After offer",
                    "churn": float(row["current_churn"]),
                }
            )
        line_chart_svg(
            rows=churn_rows,
            series_key="series",
            x_key="interaction_number",
            y_key="churn",
            title="Before vs After Churn Risk",
            subtitle="Observed churn probabilities around each recorded offer interaction.",
            y_label="Churn probability",
            output_path=churn_chart,
            x_label="Interaction number",
            series_order=["Before offer", "After offer"],
        )

        feedback_rows = []
        for _, row in feedback_df.iterrows():
            feedback_rows.append(
                {
                    "interaction_number": int(row["interaction_number"]),
                    "series": "Accepted offers",
                    "count": int(row["accepted_count"]),
                }
            )
            feedback_rows.append(
                {
                    "interaction_number": int(row["interaction_number"]),
                    "series": "Rejected offers",
                    "count": int(row["rejected_count"]),
                }
            )
        line_chart_svg(
            rows=feedback_rows,
            series_key="series",
            x_key="interaction_number",
            y_key="count",
            title="Accepted vs Rejected Offer Trend",
            subtitle="Cumulative feedback outcomes from update_history.csv.",
            y_label="Cumulative count",
            output_path=feedback_chart,
            x_label="Interaction number",
            series_order=["Accepted offers", "Rejected offers"],
        )
    else:
        churn_chart.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        feedback_chart.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")

    write_report_readme(
        {
            "feature_chart": feature_chart,
            "offer_chart": offer_chart,
            "churn_chart": churn_chart,
            "feedback_chart": feedback_chart,
            "feature_csv": feature_csv,
            "offer_csv": offer_csv,
            "feedback_csv": feedback_csv,
        }
    )
    write_index_html(
        feature_df,
        offer_df,
        feedback_df,
        {
            "feature_chart": feature_chart,
            "offer_chart": offer_chart,
            "churn_chart": churn_chart,
            "feedback_chart": feedback_chart,
            "feature_csv": feature_csv,
            "offer_csv": offer_csv,
            "feedback_csv": feedback_csv,
        },
    )

    print(f"Generated diagnostics in {OUTPUT_DIR}")
    for path in [feature_chart, offer_chart, churn_chart, feedback_chart]:
        print(path)


if __name__ == "__main__":
    main()
