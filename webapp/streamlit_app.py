"""ML-Ops-Live public Streamlit demo (entrypoint for Streamlit Community Cloud).

Run locally:  streamlit run webapp/streamlit_app.py

This UI is a read-only presentation and inference layer over the existing ML-Ops-Live
project. It never trains, promotes, rolls back, or writes to MLflow / SQLite stores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import demo_core as core  # noqa: E402

st.set_page_config(
    page_title="ML-Ops-Live | NYC Taxi Trip-Duration MLOps",
    page_icon="ðŸš•",
    layout="wide",
)

GITHUB_URL = "https://github.com/najahaja/mlops-live"


# --------------------------------------------------------------------------- cached loaders


@st.cache_resource(show_spinner="Loading champion modelâ€¦")
def get_model() -> tuple[object | None, core.ModelInfo | None, str | None]:
    try:
        pipeline, info = core.load_bundled_model()
        return pipeline, info, None
    except core.DemoModelError as exc:
        return None, None, str(exc)


@st.cache_data(show_spinner=False)
def get_asset(name: str) -> dict | None:
    return core.load_json_asset(name)


@st.cache_data(show_spinner=False)
def get_zones() -> dict[int, str]:
    return core.load_zone_lookup()


def resolve_api_url() -> str | None:
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return core.get_configured_api_url(secrets=secrets)


def fmt(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, int | float) else "n/a"


pipeline, model_info, model_error = get_model()
api_url = resolve_api_url()
metrics = get_asset("metrics.json")
holdout = get_asset("holdout_test_metrics.json")
out_of_time = get_asset("feb_2024_out_of_time_evaluation.json")
controlled = get_asset("controlled_drift_retraining_report.json")
http_load = get_asset("http_load_test_report.json")
inproc_load = get_asset("load_test_report.json")
zones = get_zones()

# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("ðŸš• ML-Ops-Live")
    st.caption("End-to-end trip-duration prediction & MLOps platform")
    st.subheader("Serving status")
    if api_url:
        st.info("Inference mode: external FastAPI service (configured via secrets).")
    elif model_info is not None:
        st.success("Inference mode: bundled champion model (in-app).")
    else:
        st.error("Model unavailable in this deployment.")
    if model_info is not None:
        st.markdown(
            f"- **Model:** `{model_info.model_name}`\n"
            f"- **Registry version:** `v{model_info.registry_version}` "
            f"(`@{model_info.registry_alias}` at export)\n"
            f"- **Integrity (SHA-256):** {'verified âœ…' if model_info.sha256_verified else 'n/a'}"
        )
    st.divider()
    st.markdown(f"[Source code on GitHub]({GITHUB_URL})")
    st.caption(
        "Public demo is read-only: no retraining, promotion, rollback, or database writes "
        "can be triggered from this page."
    )

st.title("ML-Ops-Live â€” NYC Taxi Trip-Duration Prediction & MLOps Platform")
st.markdown(
    "Predicts **NYC Green Taxi trip duration (minutes)** at ride-request time and wraps the "
    "model in a full MLOps lifecycle: MLflow registry, FastAPI serving, PSI/KS drift "
    "monitoring, gated retraining, human approval and rollback."
)

tab_overview, tab_predict, tab_perf, tab_drift, tab_lifecycle = st.tabs(
    ["Overview", "Predict", "Model performance", "Monitoring & drift", "MLOps lifecycle"]
)

# --------------------------------------------------------------------------- overview

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned trips (Jan + Feb 2024)", "102,189")
    c2.metric(
        "Holdout MAE (champion)",
        f"{fmt((holdout or {}).get('champion', {}).get('test_metrics', {}).get('mae'), 2)} min",
    )
    c3.metric(
        "Out-of-time MAE (Feb 2024)",
        f"{fmt((out_of_time or {}).get('champion_v2_hist_gbdt_metrics', {}).get('mae'), 2)} min",
    )
    c4.metric("Automated tests", "76 passing")

    st.subheader("Architecture")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb",
          fontname="Helvetica", fontsize=10];
          tlc [label="NYC TLC Parquet\\n(public data)"];
          prep [label="Ingest + clean\\nchronological split"];
          train [label="Train Dummy / Ridge /\\nHistGBDT"];
          reg [label="MLflow registry\\n@champion / @challenger"];
          api [label="FastAPI service\\n/predict /feedback /dashboard"];
          db [label="SQLite logs\\npredictions, feedback, audit"];
          drift [label="PSI + KS drift\\nweekday coverage gate"];
          retrain [label="Retrain â†’ stage\\n@challenger"];
          gate [label="Champion vs challenger\\ngate + human approval"];
          ui [label="This Streamlit demo\\n(read-only)", fillcolor="#fff4d6"];
          tlc -> prep -> train -> reg -> api -> db -> drift -> retrain -> gate -> reg;
          reg -> ui [style=dashed, label="exported\\nartifact"];
        }
        """
    )

    st.subheader("Dataset & features")
    st.markdown(
        "- **Source:** NYC Taxi & Limousine Commission Green Taxi trip records "
        "(January and February 2024).\n"
        "- **Cleaning:** 1â€“60 min duration, 0.01â€“100 mi distance, valid zone IDs "
        "(Jan: 56,551 â†’ 52,529 rows; Feb: 53,577 â†’ 49,660 rows).\n"
        "- **Split:** chronological 70/15/15 train/validation/holdout on January; "
        "February used only for out-of-time evaluation.\n"
        "- **Leakage-safe features (known at pickup):** pickup zone, drop-off zone, trip "
        "distance, passenger count, pickup hour, pickup day of week."
    )

    st.subheader("How to read the numbers on this site")
    st.markdown(
        "| Evidence type | What it means |\n"
        "| :--- | :--- |\n"
        "| **Offline evaluation** | Metrics on held-out historical data (validation, isolated "
        "holdout, full February out-of-time). |\n"
        "| **Controlled experiment** | Deliberately constructed drift scenario run in an isolated "
        "sandbox to exercise retraining and governance. Not real production drift. |\n"
        "| **Local load test** | Latency measured on a single Windows laptop (localhost). Not "
        "cloud or production capacity. |\n"
        "| **Live predictions here** | Real model inference, but **not logged** and not used "
        "for monitoring. |"
    )

# --------------------------------------------------------------------------- predict

with tab_predict:
    st.subheader("Interactive trip-duration prediction")
    st.caption(
        "Inputs are validated against the same Pydantic schema used by the FastAPI "
        "`POST /predict` endpoint."
    )

    zone_ids = sorted(zones) if zones else list(range(1, 266))

    def zone_label(zone_id: int) -> str:
        return f"{zone_id} â€” {zones[zone_id]}" if zone_id in zones else str(zone_id)

    with st.form("predict_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            pu = st.selectbox(
                "Pickup zone (TLC LocationID)",
                zone_ids,
                index=zone_ids.index(74) if 74 in zone_ids else 0,
                format_func=zone_label,
                help="NYC TLC taxi zone where the trip starts.",
            )
            do = st.selectbox(
                "Drop-off zone (TLC LocationID)",
                zone_ids,
                index=zone_ids.index(75) if 75 in zone_ids else 0,
                format_func=zone_label,
                help="NYC TLC taxi zone where the trip ends.",
            )
            distance = st.number_input(
                "Trip distance (miles)",
                min_value=0.0,
                max_value=150.0,
                value=2.5,
                step=0.1,
                help="Valid range: 0.01â€“100 miles (training data filter).",
            )
        with col_b:
            passengers = st.number_input(
                "Passenger count",
                min_value=0,
                max_value=12,
                value=1,
                step=1,
                help="Valid range: 1â€“9.",
            )
            hour = st.slider("Pickup hour (0â€“23)", 0, 23, 17)
            day = st.selectbox(
                "Pickup day of week",
                list(range(7)),
                index=2,
                format_func=lambda d: core.DAY_NAMES[d],
            )
        submitted = st.form_submit_button("Predict duration", type="primary")

    if submitted:
        features, errors = core.validate_trip_input(
            {
                "PULocationID": int(pu),
                "DOLocationID": int(do),
                "trip_distance": float(distance),
                "passenger_count": float(passengers),
                "pickup_hour": int(hour),
                "pickup_day_of_week": int(day),
            }
        )
        if errors:
            st.error("Please correct the following input(s):")
            for message in errors:
                st.markdown(f"- {message}")
        else:
            result = None
            if api_url:
                try:
                    remote = core.predict_duration_remote(api_url, features)
                    result = (
                        remote["predicted_duration_minutes"],
                        f"external FastAPI service ({remote['model_name']} "
                        f"v{remote['model_version']})",
                    )
                except core.RemoteServiceError as exc:
                    st.warning(f"{exc} Falling back to the bundled model.")
            if result is None:
                if pipeline is None:
                    st.error(model_error or "The model is unavailable in this deployment.")
                else:
                    try:
                        minutes = core.predict_duration_local(pipeline, features)
                        result = (minutes, f"bundled champion ({model_info.model_name})")
                    except core.DemoModelError as exc:
                        st.error(str(exc))
            if result is not None:
                minutes, source = result
                st.metric("Predicted trip duration", f"{minutes:.1f} minutes")
                st.caption(
                    f"Served by: {source}. Typical error on the holdout set: median absolute "
                    "error â‰ˆ 1.7 min, 90th percentile â‰ˆ 6.3 min."
                )
                if distance > 0 and minutes > 0:
                    st.caption(f"Implied average speed: {distance / (minutes / 60):.1f} mph.")

# --------------------------------------------------------------------------- performance

with tab_perf:
    st.subheader("Ridge baseline vs. HistGradientBoosting")
    rows = core.model_comparison_rows(metrics, holdout, out_of_time)
    if not rows:
        st.info("Evaluation reports are not available in this deployment.")
    else:
        if holdout:
            champ = holdout["champion"]["test_metrics"]
            chall = holdout["challenger"]["test_metrics"]
            comp = holdout.get("comparison", {})
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Holdout MAE (HistGBDT)",
                f"{champ['mae']:.3f} min",
                f"-{comp.get('mae_reduction_pct', 0):.1f}% vs Ridge",
                delta_color="inverse",
            )
            m2.metric(
                "Holdout RMSE (HistGBDT)",
                f"{champ['rmse']:.3f} min",
                f"-{comp.get('rmse_reduction_pct', 0):.1f}% vs Ridge",
                delta_color="inverse",
            )
            m3.metric(
                "Holdout RÂ² (HistGBDT)", f"{champ['r2']:.3f}", f"{champ['r2'] - chall['r2']:+.3f}"
            )

        frame = pd.DataFrame(rows)
        metric_choice = st.radio("Metric", ["MAE (min)", "RMSE (min)", "RÂ²"], horizontal=True)
        chart_frame = frame[frame["Model"] != "dummy_median"]
        st.bar_chart(
            chart_frame,
            x="Evaluation split",
            y=metric_choice,
            color="Model",
            stack=False,
            height=320,
        )
        st.dataframe(
            frame.style.format(
                {"MAE (min)": "{:.4f}", "RMSE (min)": "{:.4f}", "RÂ²": "{:.4f}"}, na_rep="n/a"
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "All metrics are offline evaluations on historical data. Validation was used for "
            "model selection; the January holdout and full February month were never used for "
            "training or selection."
        )

    st.subheader("Serving latency (local benchmarks)")
    if http_load:
        lat = http_load.get("client_latency_ms", {})
        l1, l2, l3, l4 = st.columns(4)
        l1.metric(
            "Requests OK", f"{http_load.get('success_count')}/{http_load.get('num_requests')}"
        )
        l2.metric("Throughput", f"{http_load.get('requests_per_second')} req/s")
        l3.metric("Client P50", f"{fmt(lat.get('p50'), 1)} ms")
        l4.metric("Client P95", f"{fmt(lat.get('p95'), 1)} ms")
        st.caption(
            f"Real HTTP over localhost TCP to a local Uvicorn server, concurrency "
            f"{http_load.get('concurrency')}. {http_load.get('methodology_note', '')}"
        )
    if inproc_load:
        with st.expander("In-process TestClient microbenchmark (no network)"):
            st.json(inproc_load)

# --------------------------------------------------------------------------- drift

with tab_drift:
    st.subheader("Why drift monitoring?")
    st.markdown(
        "Trip patterns change (seasonality, events, new routes). ML-Ops-Live compares the "
        "training distribution with new data using **PSI** (threshold 0.25) and the "
        "**two-sample Kolmogorovâ€“Smirnov test** (statistic > 0.1 and p < 0.01). Calendar "
        "features are only evaluated when all 7 weekdays are adequately covered; otherwise "
        "they are reported as `insufficient_data` instead of false drift."
    )

    st.subheader("Real data: January vs. full February 2024")
    feb_drift = (out_of_time or {}).get("jan_vs_feb_2024_drift")
    if not feb_drift:
        st.info("Out-of-time drift report not available in this deployment.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Overall drift detected", "Yes" if feb_drift["overall_drift_detected"] else "No")
        d2.metric("Reference rows (Jan train)", f"{feb_drift['reference_rows']:,}")
        d3.metric("Current rows (Feb)", f"{feb_drift['current_rows']:,}")
        st.dataframe(pd.DataFrame(core.drift_rows(feb_drift)), hide_index=True, width="stretch")
        st.caption(
            "Result: no feature or prediction drift between January and February 2024 "
            "(all PSI < 0.02), consistent with the stable out-of-time MAE."
        )

    st.subheader("Controlled experiment: synthetic regime shift (sandbox)")
    if not controlled:
        st.info("Controlled experiment report not available in this deployment.")
    else:
        neg = controlled["step2_negative_drift_control"]
        pos = controlled["step3_positive_drift_detection"]
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Negative control** (train vs. validation slice)")
            st.write(
                f"Drift detected: **{neg['overall_drift_detected']}** Â· retrained: "
                f"**{neg['only_if_drifted_retrained']}**"
            )
            st.caption(
                f"`pickup_day_of_week`: insufficient data â€” {neg['weekday_status_reason']}"
            )
        with e2:
            st.markdown("**Positive control** (short-haul baseline vs. long-haul trips > 3.5 mi)")
            st.write(f"Drift detected: **{pos['overall_drift_detected']}**")
            st.write(
                f"`trip_distance` PSI **{pos['trip_distance_psi']}**, "
                f"KS {pos['trip_distance_ks']}; "
                f"prediction PSI **{pos['prediction_drift_psi']}**"
            )
            st.caption("Drifted: " + ", ".join(f"`{c}`" for c in pos["drifted_columns"]))
        st.caption(
            "This shift was deliberately constructed from real January data to test the "
            "pipeline. It is not evidence of drift in live traffic."
        )

    st.subheader("Prediction logs & delayed ground-truth feedback")
    st.info(
        "The FastAPI service logs every prediction to SQLite and accepts delayed ground truth via "
        "`POST /feedback` to compute realized online MAE/RMSE. Those operational databases are "
        "local to the project and are intentionally **not published** here (privacy and data "
        "integrity). Predictions made on this page are not logged."
    )

# --------------------------------------------------------------------------- lifecycle

with tab_lifecycle:
    st.subheader("Retraining & governance workflow (read-only demonstration)")
    st.graphviz_chart(
        """
        digraph L {
          rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb",
          fontname="Helvetica", fontsize=10];
          d [label="Drift check\\n(PSI + KS)"];
          r [label="--only-if-drifted\\nretrain"];
          s [label="Stage @challenger\\nmodels/challenger/"];
          g [label="Gate: challenger MAE\\n< champion MAE"];
          a [label="Human approval\\n--approved-by"];
          p [label="Promote @champion\\n(.tmp/.bak atomic sync)"];
          b [label="Rollback\\n@previous_champion"];
          d -> r [label="drift"]; r -> s -> g -> a [label="pass"]; a -> p -> b [style=dashed];
        }
        """
    )
    if controlled:
        step4 = controlled["step4_drift_triggered_retraining"]["evaluation"]
        ch, cl, comp = step4["champion"], step4["challenger"], step4["comparison"]
        st.markdown(f"**Gate evaluation in the controlled sandbox** ({step4['eval_rows']:,} rows)")
        g1, g2, g3 = st.columns(3)
        g1.metric(f"Champion v{ch['version']} MAE", f"{ch['metrics']['mae']:.2f} min")
        g2.metric(
            f"Retrained challenger v{cl['version']} MAE",
            f"{cl['metrics']['mae']:.2f} min",
            f"-{comp['mae_reduction_pct']:.1f}%",
            delta_color="inverse",
        )
        g3.metric("Gate passed", "Yes" if step4["gate_passed"] else "No")

        gov = controlled["step5_governance_approval_and_rollback"]
        st.markdown("**Audit trail recorded in the sandbox**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Event": gov["approval"]["event_type"],
                        "Champion before": f"v{gov['approval']['previous_champion_version']}",
                        "Champion after": f"v{gov['approval']['new_champion_version']}",
                        "Gate passed": gov["approval"]["gate_passed"],
                    },
                    {
                        "Event": gov["rollback"]["event_type"],
                        "Champion before": f"v{gov['rollback']['previous_champion_version']}",
                        "Champion after": f"v{gov['rollback']['new_champion_version']}",
                        "Gate passed": gov["rollback"]["gate_passed"],
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        events = controlled.get("sqlite_records_logged_in_sandbox", {}).get(
            "lifecycle_audit_events", []
        )
        if events:
            st.caption("Logged lifecycle events: " + " â†’ ".join(f"`{e}`" for e in events))
    st.warning(
        "Retraining, promotion, approval and rollback are CLI operations that require local "
        "access to the MLflow registry. They are deliberately **not** exposed in this public app."
    )
    st.code(
        "python scripts/retrain_and_evaluate.py --only-if-drifted\n"
        "python scripts/manage_model.py --evaluate --eval-file data/processed/val.parquet\n"
        "python scripts/manage_model.py --approve --approved-by <reviewer>\n"
        "python scripts/manage_model.py --rollback-previous --approved-by <reviewer>",
        language="bash",
    )

st.divider()
st.caption(
    "Data: NYC Taxi & Limousine Commission trip record data. Built with scikit-learn, MLflow, "
    "FastAPI, SQLAlchemy and Streamlit."
)
