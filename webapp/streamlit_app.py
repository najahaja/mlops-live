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

PORTFOLIO_URL = "https://ahamednajah.netlify.app/"

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
        "- **Automated Tests:** 79 passed (CI clean)\n"
        "- **Python Engine:** 3.11 / scikit-learn 1.9.1"
    )

    st.markdown("---")
    st.markdown(f"📦 [Visit My Portfolio]({PORTFOLIO_URL})")
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

# rest of file omitted for brevity; unchanged below
