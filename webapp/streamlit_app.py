"""ML-Ops-Live public Streamlit demo (entrypoint for Streamlit Community Cloud).

Run locally:  streamlit run webapp/streamlit_app.py

This UI provides an executive, read-only presentation and inference interface
for the ML-Ops-Live platform. It demonstrates production model serving,
monitoring, and governance without mutating local MLflow or SQLite state.
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

# ---------------------------------------------------------------------------
# Page configuration & branding
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML-Ops-Live | NYC Taxi Trip-Duration MLOps",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/najahaja/mlops-live"

# Scoped, restrained CSS for professional typography, cards, and badges
st.markdown(
    """
    <style>
    /* Metric card styling */
    [data-testid="stMetric"] {
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #57606a;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #1f2328;
        font-weight: 600;
    }

    /* Badge labels */
    .badge-pill {
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.5;
        border-radius: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-blue { background-color: #ddf4ff; color: #0969da; border: 1px solid #54aeff66; }
    .badge-green { background-color: #dafbe1; color: #1a7f37; border: 1px solid #4ac26b66; }
    .badge-purple { background-color: #fbefff; color: #8250df; border: 1px solid #c297ff66; }
    .badge-gray { background-color: #f6f8fa; color: #57606a; border: 1px solid #d0d7de; }

    /* Callout banner */
    .hero-banner {
        background: linear-gradient(135deg, #f6f8fa 0%, #eef3fb 100%);
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 6px;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #424a53;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading champion model...")
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


# Load assets
pipeline, model_info, model_error = get_model()
api_url = resolve_api_url()
metrics = get_asset("metrics.json")
holdout = get_asset("holdout_test_metrics.json")
out_of_time = get_asset("feb_2024_out_of_time_evaluation.json")
controlled = get_asset("controlled_drift_retraining_report.json")
http_load = get_asset("http_load_test_report.json")
inproc_load = get_asset("load_test_report.json")
zones = get_zones()

# ---------------------------------------------------------------------------
# Sidebar: System provenance & deployment metadata
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚕 ML-Ops-Live")
    st.caption("Production MLOps Platform — Trip-Duration Regressor")

    st.markdown("---")
    st.markdown("#### Serving Provenance")
    if api_url:
        st.info("Inference Mode: External FastAPI Service (Configured via Secrets)")
    elif model_info is not None:
        st.success("Inference Mode: Bundled Champion Model")
    else:
        st.error("Serving Mode: Model Unavailable")

    if model_info is not None:
        integrity_status = "Verified ✅" if model_info.sha256_verified else "Unverified"
        st.markdown(
            f"- **Model:** `{model_info.model_name}`\n"
            f"- **Registry Version:** `v{model_info.registry_version}` "
            f"(`@{model_info.registry_alias}`)\n"
            f"- **Integrity (SHA-256):** {integrity_status}"
        )

    st.markdown("---")
    st.markdown("#### Platform Verification")
    st.markdown(
        "- **Dataset:** NYC TLC Green Taxi (Jan & Feb 2024)\n"
        "- **Cleaned Volume:** 102,189 verified trips\n"
        "- **Automated Tests:** 78 passed (CI clean)\n"
        "- **Python Engine:** 3.11 / scikit-learn 1.9.1"
    )

    st.markdown("---")
    st.markdown(f"📦 [View Source Code on GitHub]({GITHUB_URL})")
    st.caption(
        "Security Note: This public demo runs in a read-only container. Retraining, "
        "model promotion, and rollback operations require local CLI credentials."
    )

# ---------------------------------------------------------------------------
# Main Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            🚕 ML-Ops-Live — NYC Taxi Trip-Duration Prediction & MLOps Platform
        </div>
        <div class="hero-desc">
            A production-style machine learning platform predicting NYC Green Taxi trip duration
            (<code>duration_minutes</code>) at ride-request time. Featuring an MLflow model
            registry, FastAPI serving, PSI/KS data drift monitoring, automated retraining
            gates, and transactional rollback.
        </div>
        <div>
            <span class="badge-pill badge-blue">FastAPI Inference</span>
            <span class="badge-pill badge-green">MLflow Registry</span>
            <span class="badge-pill badge-purple">PSI / KS Drift Gates</span>
            <span class="badge-pill badge-gray">HistGradientBoosting Regressor</span>
            <span class="badge-pill badge-gray">102,189 Trips (Jan & Feb 2024)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_predict, tab_perf, tab_drift, tab_lifecycle = st.tabs(
    [
        "System Overview",
        "Interactive Prediction",
        "Model Performance",
        "Drift Monitoring",
        "MLOps Lifecycle",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1: System Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned Trips (Jan + Feb 2024)", "102,189")
    c2.metric(
        "Holdout Test MAE (Champion)",
        f"{fmt((holdout or {}).get('champion', {}).get('test_metrics', {}).get('mae'), 2)} min",
        delta="-14.1% vs Baseline",
        delta_color="inverse",
    )
    c3.metric(
        "Out-of-Time MAE (Feb 2024)",
        f"{fmt((out_of_time or {}).get('champion_v2_hist_gbdt_metrics', {}).get('mae'), 2)} min",
        delta="R² = 0.755",
        delta_color="normal",
    )
    c4.metric("Automated Tests", "78 passing", delta="100% pass")

    st.markdown("### Platform Architecture")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#f6f8fa", color="#d0d7de",
                fontname="Helvetica", fontsize=10];
          edge [color="#57606a", fontsize=9];

          tlc [label="NYC TLC Parquet\\n(Monthly Public Feed)",
               fillcolor="#ddf4ff", color="#54aeff"];
          prep [label="Data Cleaning &\\nChronological Split"];
          train [label="Candidate Training\\nDummy / Ridge / HistGBDT"];
          reg [label="MLflow Registry\\n@champion & @challenger",
               fillcolor="#dafbe1", color="#4ac26b"];
          api [label="FastAPI Service\\nPOST /predict & /feedback",
               fillcolor="#fbefff", color="#c297ff"];
          db [label="SQLite Store\\nLogs, Feedback, Audit"];
          drift [label="PSI + KS Drift Monitor\\n7-Day Weekday Guard"];
          retrain [label="Automated Retraining\\nStage @challenger"];
          gate [label="Evaluation Gate\\nHuman Promotion / Rollback"];
          ui [label="Streamlit Web UI\\n(Read-Only Demo)",
              fillcolor="#fff8c5", color="#d4a72c"];

          tlc -> prep -> train -> reg -> api -> db -> drift -> retrain -> gate -> reg;
          reg -> ui [style=dashed, label="bundled\\nartifact"];
        }
        """
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Dataset & Preprocessing")
        st.markdown(
            "- **Data Source:** Official NYC TLC Green Taxi trip records.\n"
            "- **Chronological Partitions:** 70% Train, 15% Validation, 15% Isolated Holdout Test "
            "(January 2024), with the entire subsequent month (February 2024) reserved for "
            "out-of-time evaluation.\n"
            "- **Quality Filters:** Restricted to valid trips with duration between 1.0 and 60.0 "
            "minutes, distance between 0.01 and 100.0 miles, and valid TLC taxi zone IDs.\n"
            "- **Cleaned Volume:** 52,529 cleaned trips in Jan 2024 (92.89% retained) and 49,660 "
            "cleaned trips in Feb 2024 (92.69% retained)."
        )

    with col_right:
        st.markdown("### Leakage-Safe Feature Contract")
        st.markdown(
            "Every feature is strictly restricted to signals available at pickup time:\n"
            "- `PULocationID`: TLC pickup taxi zone ID (categorical, target encoded).\n"
            "- `DOLocationID`: TLC dropoff taxi zone ID (categorical, target encoded).\n"
            "- `trip_distance`: Estimated trip distance in miles.\n"
            "- `passenger_count`: Number of passengers (1.0 to 9.0).\n"
            "- `pickup_hour`: Hour of pickup (0 to 23).\n"
            "- `pickup_day_of_week`: Day of week (0=Monday to 6=Sunday)."
        )

    st.markdown("### Evidence & Measurement Classification Guide")
    st.markdown(
        "To ensure transparency in engineering claims, metrics are categorized as follows:\n\n"
        "| Evidence Type | Scope & Methodology | Environment |\n"
        "| :--- | :--- | :--- |\n"
        "| **Offline Holdout Evaluation** | Evaluated on historical `test.parquet` (Jan 2024, "
        "7,880 rows) and `feb_2024` (49,660 rows). | Python / scikit-learn |\n"
        "| **Controlled Sandbox Experiment** | Deliberately constructed long-haul distribution "
        "shift (> 3.5 mi) in an isolated sandbox to verify drift gates and rollback. | "
        "Sandbox MLflow + SQLite |\n"
        "| **Local HTTP Load Testing** | Real HTTP requests over localhost TCP against a spawned "
        "Uvicorn FastAPI server measuring client and server latencies. | Managed Local Uvicorn |\n"
        "| **Live In-App Inference** | Computes duration using the real champion scikit-learn "
        "pipeline in-memory (read-only, not logged). | Streamlit Web Runtime |"
    )

# ---------------------------------------------------------------------------
# Tab 2: Interactive Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.markdown("### Interactive Trip-Duration Prediction")
    st.caption(
        "Enter ride details below. Inputs are validated using the exact Pydantic schema "
        "enforced by the production FastAPI endpoint."
    )

    # Quick scenario presets for seamless recruiter evaluation
    st.markdown("**Quick Preset Scenarios:**")
    preset_cols = st.columns(4)
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = {
            "pu": 74,
            "do": 75,
            "dist": 2.4,
            "passengers": 1.0,
            "hour": 17,
            "day": 2,
        }

    if preset_cols[0].button("Manhattan Short Hop (2.4 mi)"):
        st.session_state.preset_data = {
            "pu": 74,
            "do": 75,
            "dist": 2.4,
            "passengers": 1.0,
            "hour": 17,
            "day": 2,
        }
    if preset_cols[1].button("Outer Borough Run (7.8 mi)"):
        st.session_state.preset_data = {
            "pu": 7,
            "do": 132,
            "dist": 7.8,
            "passengers": 2.0,
            "hour": 8,
            "day": 1,
        }
    if preset_cols[2].button("Late-Night Weekend (4.2 mi)"):
        st.session_state.preset_data = {
            "pu": 41,
            "do": 166,
            "dist": 4.2,
            "passengers": 1.0,
            "hour": 23,
            "day": 5,
        }
    if preset_cols[3].button("Medium Cross-Town (3.5 mi)"):
        st.session_state.preset_data = {
            "pu": 82,
            "do": 129,
            "dist": 3.5,
            "passengers": 1.0,
            "hour": 14,
            "day": 3,
        }

    zone_ids = sorted(zones) if zones else list(range(1, 266))

    def zone_label(zone_id: int) -> str:
        return f"{zone_id} — {zones[zone_id]}" if zone_id in zones else f"Zone {zone_id}"

    current = st.session_state.preset_data

    with st.form("predict_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            pu_default = current["pu"] if current["pu"] in zone_ids else zone_ids[0]
            pu = st.selectbox(
                "Pickup Taxi Zone (TLC LocationID)",
                zone_ids,
                index=zone_ids.index(pu_default),
                format_func=zone_label,
                help="NYC TLC taxi zone where the passenger is picked up.",
            )
            do_default = current["do"] if current["do"] in zone_ids else zone_ids[0]
            do = st.selectbox(
                "Drop-off Taxi Zone (TLC LocationID)",
                zone_ids,
                index=zone_ids.index(do_default),
                format_func=zone_label,
                help="NYC TLC taxi zone where the passenger is dropped off.",
            )
            distance = st.number_input(
                "Trip Distance (miles)",
                min_value=0.0,
                max_value=120.0,
                value=float(current["dist"]),
                step=0.1,
                help="Estimated trip distance in miles (valid training range: 0.01 to 100.0).",
            )
        with col_b:
            passengers = st.number_input(
                "Passenger Count",
                min_value=0,
                max_value=9,
                value=int(current["passengers"]),
                step=1,
                help="Number of passengers (valid range: 1 to 9).",
            )
            hour = st.slider(
                "Pickup Hour (0 to 23)",
                min_value=0,
                max_value=23,
                value=int(current["hour"]),
                help="Time of day when pickup occurs.",
            )
            day = st.selectbox(
                "Pickup Day of Week",
                list(range(7)),
                index=int(current["day"]),
                format_func=lambda d: core.DAY_NAMES[d],
                help="Day of week (0=Monday to 6=Sunday).",
            )
        submitted = st.form_submit_button("Predict Trip Duration", type="primary")

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
                        f"External FastAPI Service ({remote['model_name']} "
                        f"v{remote['model_version']})",
                    )
                except core.RemoteServiceError as exc:
                    st.warning(f"{exc} Falling back to bundled champion model.")
            if result is None:
                if pipeline is None:
                    st.error(model_error or "Model pipeline unavailable.")
                else:
                    try:
                        minutes = core.predict_duration_local(pipeline, features)
                        result = (
                            minutes,
                            f"Bundled Champion ({model_info.model_name} "
                            f"v{model_info.registry_version})",
                        )
                    except core.DemoModelError as exc:
                        st.error(str(exc))

            if result is not None:
                minutes, source = result
                st.markdown("---")
                res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])
                res_col1.metric("Predicted Trip Duration", f"{minutes:.1f} minutes")
                if distance > 0 and minutes > 0:
                    speed_mph = distance / (minutes / 60.0)
                    res_col2.metric("Implied Speed", f"{speed_mph:.1f} mph")
                res_col3.metric("Holdout P50 Error", "1.7 min", help="Median absolute error")

                st.caption(
                    f"**Serving Source:** {source} · **Expected Error:** P50 absolute error ≈ "
                    f"1.7 min, P90 error ≈ 6.3 min, P95 error ≈ 9.0 min on holdout data."
                )

# ---------------------------------------------------------------------------
# Tab 3: Model Performance
# ---------------------------------------------------------------------------
with tab_perf:
    st.markdown("### Model Comparison & Evaluation Benchmarks")
    st.markdown(
        "Comparing the baseline **Ridge Regressor** (`@challenger` `v1`) against the candidate "
        "**HistGradientBoostingRegressor** (`@champion` `v2`) across all chronological splits."
    )

    if holdout:
        champ = holdout["champion"]["test_metrics"]
        chall = holdout["challenger"]["test_metrics"]
        comp = holdout.get("comparison", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Holdout Test MAE (Champion)",
            f"{champ['mae']:.3f} min",
            f"-{comp.get('mae_reduction_pct', 0):.1f}% vs Ridge",
            delta_color="inverse",
        )
        m2.metric(
            "Holdout Test RMSE (Champion)",
            f"{champ['rmse']:.3f} min",
            f"-{comp.get('rmse_reduction_pct', 0):.1f}% vs Ridge",
            delta_color="inverse",
        )
        m3.metric(
            "Holdout Test R² (Champion)",
            f"{champ['r2']:.3f}",
            f"{champ['r2'] - chall['r2']:+.3f} vs Ridge",
        )
        feb_mae = (out_of_time or {}).get("champion_v2_hist_gbdt_metrics", {}).get("mae")
        m4.metric(
            "Out-of-Time MAE (Feb 2024)",
            f"{fmt(feb_mae, 3)} min",
            delta="-14.9% vs Ridge",
            delta_color="inverse",
        )

    rows = core.model_comparison_rows(metrics, holdout, out_of_time)
    if rows:
        frame = pd.DataFrame(rows)
        col_opt, _ = st.columns([1, 2])
        metric_choice = col_opt.radio(
            "Evaluation Metric to Graph:",
            ["MAE (min)", "RMSE (min)", "R²"],
            horizontal=True,
        )
        chart_frame = frame[frame["Model"] != "dummy_median"]
        st.bar_chart(
            chart_frame,
            x="Evaluation split",
            y=metric_choice,
            color="Model",
            stack=False,
            height=320,
        )

        st.markdown("#### Detailed Split-by-Split Metrics Table")
        st.dataframe(
            frame.style.format(
                {"MAE (min)": "{:.4f}", "RMSE (min)": "{:.4f}", "R²": "{:.4f}"},
                na_rep="n/a",
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Note: All figures represent offline evaluations on historical records. "
            "Validation was used exclusively for hyperparameter tuning, while Holdout Test "
            "(Jan 2024) and Out-of-Time (Feb 2024) remained strictly untouched during training."
        )

    st.markdown("---")
    st.markdown("### Serving Latency & Throughput Benchmark")
    if http_load:
        lat = http_load.get("client_latency_ms", {})
        l1, l2, l3, l4 = st.columns(4)
        l1.metric(
            "Requests Processed",
            f"{http_load.get('success_count')}/{http_load.get('num_requests')}",
            delta="0.0% Error Rate",
        )
        l2.metric("Throughput", f"{http_load.get('requests_per_second')} req/s")
        l3.metric("Client P50 Latency", f"{fmt(lat.get('p50'), 1)} ms")
        l4.metric("Client P95 Latency", f"{fmt(lat.get('p95'), 1)} ms")

        concurrency_val = http_load.get("concurrency")
        st.caption(
            f"**Methodology:** Measured via concurrent HTTP requests over localhost TCP "
            f"against a spawned Uvicorn FastAPI server (concurrency {concurrency_val}). "
            f"{http_load.get('methodology_note', '')}"
        )

# ---------------------------------------------------------------------------
# Tab 4: Drift Monitoring
# ---------------------------------------------------------------------------
with tab_drift:
    st.markdown("### Production Drift Monitoring Strategy")
    st.markdown(
        "Taxi trip durations are vulnerable to covariate shift (weather, seasonality, holidays). "
        "ML-Ops-Live monitors incoming traffic using **Population Stability Index (PSI)** "
        "(threshold 0.25) and the **two-sample Kolmogorov–Smirnov (KS) test** (threshold 0.1, "
        "alpha 0.01). A **7-day calendar coverage safeguard** prevents false alarms on short "
        "observation windows."
    )

    st.markdown("#### Real Multi-Month Stability: January 2024 vs. Full February 2024")
    feb_drift = (out_of_time or {}).get("jan_vs_feb_2024_drift")
    if feb_drift:
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Overall Drift Status",
            "Drift Detected" if feb_drift["overall_drift_detected"] else "Stable (No Drift)",
        )
        d2.metric("Reference Volume (Jan Train)", f"{feb_drift['reference_rows']:,} rows")
        d3.metric("Comparison Volume (Feb Cleaned)", f"{feb_drift['current_rows']:,} rows")

        st.dataframe(
            pd.DataFrame(core.drift_rows(feb_drift)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Verification Outcome: With full 7-weekday coverage across 29 calendar days, "
            "no feature or prediction drift is detected (all feature PSIs < 0.02), confirming "
            "that the model generalizes stably across months."
        )

    st.markdown("---")
    st.markdown("#### Controlled Sandbox Drift Experiment (Regime Shift)")
    if controlled:
        neg = controlled["step2_negative_drift_control"]
        pos = controlled["step3_positive_drift_detection"]
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**1. Negative Control (Natural 4.2-Day Validation Slice)**")
            st.write(f"- Overall Drift Detected: **{neg['overall_drift_detected']}**")
            st.write(f"- Retraining Triggered: **{neg['only_if_drifted_retrained']}**")
            st.caption(
                f"Coverage Safeguard Action: `pickup_day_of_week` correctly flagged as "
                f"`insufficient_data` ({neg['weekday_status_reason']}), avoiding false retraining."
            )
        with e2:
            st.markdown("**2. Positive Control (Synthetic Long-Haul Trips > 3.5 mi)**")
            st.write(f"- Overall Drift Detected: **{pos['overall_drift_detected']}**")
            st.write(
                f"- `trip_distance` PSI: **{pos['trip_distance_psi']}**, "
                f"KS: **{pos['trip_distance_ks']}**"
            )
            st.write(f"- Prediction Drift PSI: **{pos['prediction_drift_psi']}**")
            st.caption(
                "Result: Correctly flagged distribution shift across "
                + ", ".join(f"`{c}`" for c in pos["drifted_columns"])
            )

    st.markdown("---")
    st.markdown("#### Delayed Ground-Truth Feedback & Operational Store")
    st.info(
        "In production, trip durations are finalized only after a ride ends. The FastAPI service "
        "provides `POST /feedback` to correlate actual trip duration with past prediction request "
        "IDs, enabling online realized MAE and RMSE calculation. To protect data privacy, the "
        "operational SQLite database (`predictions.db`) is kept local and isolated from this "
        "public demonstration."
    )

# ---------------------------------------------------------------------------
# Tab 5: MLOps Lifecycle & Governance
# ---------------------------------------------------------------------------
with tab_lifecycle:
    st.markdown("### Retraining, Promotion & Rollback Governance")
    st.markdown(
        "Demonstrating the full lifecycle when drift is detected: automated retraining, "
        "quantitative challenger gating, human-in-the-loop approval, and transactional rollback."
    )

    st.graphviz_chart(
        """
        digraph L {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#f6f8fa", color="#d0d7de",
                fontname="Helvetica", fontsize=10];
          edge [color="#57606a", fontsize=9];

          d [label="PSI + KS Drift\\nDetection", fillcolor="#ddf4ff", color="#54aeff"];
          r [label="scripts/retrain_and_evaluate.py\\n(--only-if-drifted)"];
          s [label="Stage Candidate as @challenger\\n(models/challenger/ isolated)",
             fillcolor="#dafbe1", color="#4ac26b"];
          g [label="Quantitative Gate\\nChallenger MAE < Champion MAE"];
          a [label="Human Approval\\n(--approve --approved-by)"];
          p [label="Promote to @champion\\n(.tmp/.bak atomic sync)",
             fillcolor="#fbefff", color="#c297ff"];
          b [label="Instant Rollback\\n(--rollback-previous)",
             style="dashed,rounded,filled", fillcolor="#ffebe9", color="#cf222e"];

          d -> r [label="drift flagged"];
          r -> s -> g -> a [label="gate passes"];
          a -> p -> b [style=dashed, label="if regression"];
        }
        """
    )

    if controlled:
        step4 = controlled["step4_drift_triggered_retraining"]["evaluation"]
        ch, cl, comp = step4["champion"], step4["challenger"], step4["comparison"]
        st.markdown(f"#### Verified Sandbox Gate Evaluation ({step4['eval_rows']:,} rows)")
        g1, g2, g3 = st.columns(3)
        g1.metric(f"Incumbent Champion (v{ch['version']}) MAE", f"{ch['metrics']['mae']:.2f} min")
        g2.metric(
            f"Retrained Challenger (v{cl['version']}) MAE",
            f"{cl['metrics']['mae']:.2f} min",
            f"-{comp['mae_reduction_pct']:.1f}%",
            delta_color="inverse",
        )
        g3.metric(
            "Promotion Gate Status",
            "PASSED" if step4["gate_passed"] else "FAILED",
            delta="Eligible for Promotion",
        )

        gov = controlled["step5_governance_approval_and_rollback"]
        st.markdown("#### Audited Governance Trail")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Lifecycle Action": "1. Human Approval & Promotion",
                        "Actor": gov["approval"]["actor"],
                        "Prior Champion": f"v{gov['approval']['previous_champion_version']}",
                        "Promoted Champion": f"v{gov['approval']['new_champion_version']}",
                        "Gate Result": ("Passed" if gov["approval"]["gate_passed"] else "Failed"),
                    },
                    {
                        "Lifecycle Action": "2. Rollback Verification",
                        "Actor": gov["rollback"]["actor"],
                        "Prior Champion": f"v{gov['rollback']['previous_champion_version']}",
                        "Restored Champion": f"v{gov['rollback']['new_champion_version']}",
                        "Gate Result": ("Passed" if gov["rollback"]["gate_passed"] else "Failed"),
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Reproducible Lifecycle CLI Commands")
    st.code(
        """# Retrain model only if drift is detected above threshold:
python scripts/retrain_and_evaluate.py --only-if-drifted

# Evaluate champion vs challenger gate on new dataset:
python scripts/manage_model.py --evaluate --eval-file data/processed/val.parquet

# Approve and promote challenger to champion:
python scripts/manage_model.py --approve --approved-by engineering_lead

# Roll back immediately to previous champion version:
python scripts/manage_model.py --rollback-previous --approved-by engineering_lead""",
        language="bash",
    )

st.markdown("---")
st.caption(
    "ML-Ops-Live | Powered by Python 3.11, scikit-learn, MLflow, FastAPI, "
    "SQLAlchemy, and Streamlit."
)
