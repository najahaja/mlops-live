"""Monitoring dashboard aggregation and HTML rendering for FastAPI."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import Engine

from mlops_live.serving.database import count_logged_predictions, fetch_recent_predictions


def build_dashboard_summary(
    engine: Engine,
    active_model_name: str,
    active_model_version: str,
    active_model_alias: str,
    active_model_source: str,
    recent_limit: int = 50,
) -> dict[str, Any]:
    """Aggregate live serving, drift, and governance metrics from SQLite."""
    from mlops_live.lifecycle.workflow import fetch_lifecycle_audit_logs
    from mlops_live.monitoring.drift import fetch_recent_drift_reports

    total_predictions = count_logged_predictions(engine)
    recent_preds = fetch_recent_predictions(engine, limit=recent_limit)
    drift_history = fetch_recent_drift_reports(engine, limit=10)
    audit_logs = fetch_lifecycle_audit_logs(engine, limit=20)

    if recent_preds:
        latencies = np.array([float(r["latency_ms"]) for r in recent_preds], dtype=np.float64)
        durations = np.array(
            [float(r["predicted_duration_minutes"]) for r in recent_preds],
            dtype=np.float64,
        )
        latency_stats = {
            "mean_ms": round(float(np.mean(latencies)), 3),
            "p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "max_ms": round(float(np.max(latencies)), 3),
        }
        prediction_stats = {
            "mean_duration_min": round(float(np.mean(durations)), 3),
            "median_duration_min": round(float(np.median(durations)), 3),
        }
    else:
        latency_stats = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        prediction_stats = {"mean_duration_min": 0.0, "median_duration_min": 0.0}

    return {
        "active_model": {
            "model_name": active_model_name,
            "model_version": active_model_version,
            "model_alias": active_model_alias,
            "model_source": active_model_source,
        },
        "total_predictions_logged": total_predictions,
        "latency_stats": latency_stats,
        "prediction_stats": prediction_stats,
        "recent_predictions": recent_preds[:15],
        "drift_reports": drift_history,
        "lifecycle_audit_logs": audit_logs,
    }


def render_dashboard_html(summary: dict[str, Any]) -> str:
    """Render a lightweight self-contained HTML monitoring dashboard."""
    am = summary["active_model"]
    lat = summary["latency_stats"]
    pred_stats = summary["prediction_stats"]

    drift_rows_html = ""
    for d in summary["drift_reports"]:
        badge = "DRIFT" if d["overall_drift_detected"] else "STABLE"
        color = "#b91c1c" if d["overall_drift_detected"] else "#15803d"
        cols = ", ".join(d["drifted_columns"]) if d["drifted_columns"] else "None"
        drift_rows_html += (
            f"<tr><td>{d['created_at_utc']}</td>"
            f"<td>{d['current_source']} ({d['current_rows']} rows)</td>"
            f"<td>{d['max_psi']:.4f}</td><td>{d['max_ks_stat']:.4f}</td>"
            f"<td style='color:{color};font-weight:bold'>{badge}</td>"
            f"<td>{cols}</td></tr>"
        )
    if not drift_rows_html:
        drift_rows_html = "<tr><td colspan='6'>No drift reports logged yet.</td></tr>"

    audit_rows_html = ""
    for a in summary["lifecycle_audit_logs"]:
        audit_rows_html += (
            f"<tr><td>{a['created_at_utc']}</td><td><code>{a['event_type']}</code></td>"
            f"<td>{a['actor']}</td><td>v{a['previous_champion_version']} &rarr; "
            f"v{a['new_champion_version']}</td><td>v{a['challenger_version']}</td>"
            f"<td>{a['notes']}</td></tr>"
        )
    if not audit_rows_html:
        audit_rows_html = "<tr><td colspan='6'>No lifecycle governance events logged yet.</td></tr>"

    pred_rows_html = ""
    for p in summary["recent_predictions"]:
        pred_rows_html += (
            f"<tr><td>{p['created_at_utc']}</td><td><code>{p['request_id'][:8]}</code></td>"
            f"<td>{p['PULocationID']} &rarr; {p['DOLocationID']}</td>"
            f"<td>{p['trip_distance']:.2f} mi</td>"
            f"<td><b>{p['predicted_duration_minutes']:.2f} min</b></td>"
            f"<td>{p['latency_ms']:.2f} ms</td>"
            f"<td>{p['model_name']} (v{p['model_version']})</td></tr>"
        )
    if not pred_rows_html:
        pred_rows_html = "<tr><td colspan='7'>No predictions logged yet.</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ML-Ops-Live Monitoring Dashboard</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      margin: 24px; background: #f8fafc; color: #0f172a;
    }}
    h1, h2 {{ margin-bottom: 8px; }}
    .cards {{
      display: grid; gap: 14px; margin: 16px 0;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    }}
    .card {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; }}
    .card .label {{ font-size: 12px; color: #475569; text-transform: uppercase; }}
    .card .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    table {{
      width: 100%; border-collapse: collapse; background: #fff;
      border: 1px solid #cbd5e1; margin-bottom: 24px; font-size: 13px;
    }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>ML-Ops-Live — Production MLOps Dashboard</h1>
  <p>NYC Green Taxi Trip Duration Service &bull; Native Local SQLite &amp; MLflow Registry</p>
  <div class="cards">
    <div class="card">
      <div class="label">Active Model (@{am["model_alias"]})</div>
      <div class="val">{am["model_name"]} (v{am["model_version"]})</div>
    </div>
    <div class="card">
      <div class="label">Total Predictions Logged</div>
      <div class="val">{summary["total_predictions_logged"]}</div>
    </div>
    <div class="card">
      <div class="label">Latency (Mean / P95)</div>
      <div class="val">{lat["mean_ms"]} / {lat["p95_ms"]} ms</div>
    </div>
    <div class="card">
      <div class="label">Predicted Duration (Mean)</div>
      <div class="val">{pred_stats["mean_duration_min"]} min</div>
    </div>
  </div>
  <h2>Data &amp; Prediction Drift History (PSI &amp; KS)</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp (UTC)</th><th>Window</th><th>Max PSI</th>
        <th>Max KS</th><th>Status</th><th>Drifted Columns</th>
      </tr>
    </thead>
    <tbody>{drift_rows_html}</tbody>
  </table>
  <h2>Model Governance &amp; Lifecycle Audit Log</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp (UTC)</th><th>Event</th><th>Actor</th>
        <th>Champion Transition</th><th>Challenger</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>{audit_rows_html}</tbody>
  </table>
  <h2>Recent Live Predictions</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp (UTC)</th><th>Req ID</th><th>Route (PU &rarr; DO)</th>
        <th>Distance</th><th>Predicted ETA</th><th>Latency</th><th>Model</th>
      </tr>
    </thead>
    <tbody>{pred_rows_html}</tbody>
  </table>
</body>
</html>"""
