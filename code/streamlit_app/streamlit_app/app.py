
"""
Powerlifting Deadlift Predictor — Home / Prediction page.
Run locally with:  streamlit run app.py
"""
 
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
# ----------------------------------------------------------------------
# Resolve paths relative to THIS file, not the current working directory.
# Streamlit Cloud does not guarantee the working directory is this
# script's own folder, so a bare "final_model.pkl" can silently fail
# while running "streamlit run app.py" from inside this folder locally
# works by coincidence. Anchoring to __file__ fixes it in both places.
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
 
# ----------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deadlift Predictor",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ----------------------------------------------------------------------
# Shared style — one small CSS block reused by every page
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.5rem;}
    #MainMenu, footer {visibility: hidden;}
 
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.4rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ff6b35;
        line-height: 1.1;
    }
    .result-label {
        font-size: 0.95rem;
        color: #c9c9d9;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .info-pill {
        display: inline-block;
        background: #f3f4f6;
        border-radius: 999px;
        padding: 0.25rem 0.9rem;
        font-size: 0.82rem;
        color: #374151;
        margin: 0.15rem;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
 
# ----------------------------------------------------------------------
# Cached loaders — the model and reference sample load once per session
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "final_model.pkl")
 
 
@st.cache_data
def load_reference_data():
    return pd.read_csv(APP_DIR / "lifting_data_sample.csv")
 
 
model = load_model()
ref_data = load_reference_data()
 
# ----------------------------------------------------------------------
# Sidebar — inputs live here so the main area is fully devoted to results
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏋️ Lifter Details")
    st.caption("Enter today's squat and bench results to estimate the deadlift.")
 
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    equipment = st.selectbox(
        "Equipment", ["Raw", "Wraps", "Single-ply", "Multi-ply"],
        help="Type of supportive gear worn during the lift.",
    )
    bodyweight = st.number_input("Bodyweight (kg)", min_value=30.0, max_value=250.0, value=80.0, step=0.5)
    squat = st.number_input("Best Squat (kg)", min_value=0.0, max_value=500.0, value=150.0, step=2.5)
    bench = st.number_input("Best Bench (kg)", min_value=0.0, max_value=350.0, value=100.0, step=2.5)
 
    st.markdown("---")
    predict_clicked = st.button("Predict Deadlift", type="primary", use_container_width=True)
 
# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.markdown('<div class="app-title">Powerlifting Deadlift Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Predicts a lifter\'s best deadlift from their squat, '
    'bench, bodyweight, sex and equipment — powered by a tuned XGBoost model.</div>',
    unsafe_allow_html=True,
)
 
if not predict_clicked:
    st.info("Fill in the details on the left and click **Predict Deadlift** to see a result.")
    st.markdown("#### What this model uses")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label in zip(
        [c1, c2, c3, c4, c5],
        ["Best Squat", "Best Bench", "Bodyweight", "Sex", "Equipment"],
    ):
        col.markdown(f'<div class="info-pill">{label}</div>', unsafe_allow_html=True)
    st.caption(
        "Squat and bench are used because, in a real meet, both lifts happen before the "
        "deadlift — so their results are genuinely known at prediction time."
    )
else:
    input_row = pd.DataFrame(
        [{
            "BestSquatKg": squat,
            "BestBenchKg": bench,
            "BodyweightKg": bodyweight,
            "Sex": sex,
            "Equipment": equipment,
        }]
    )
    prediction = float(model.predict(input_row)[0])
 
    left, right = st.columns([1, 1.4])
 
    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Best Deadlift</div>
                <div class="result-value">{prediction:.1f} kg</div>
                <div class="result-label">≈ {prediction * 2.20462:.1f} lb</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Typical prediction error for this model is about ±14.9 kg (MAE on test data) — "
            "treat this as an informed estimate, not an exact figure."
        )
 
        # percentile vs similar lifters (same sex + equipment) in the reference sample
        peer_group = ref_data[(ref_data["Sex"] == sex) & (ref_data["Equipment"] == equipment)]
        if len(peer_group) > 20:
            percentile = (peer_group["BestDeadliftKg"] < prediction).mean() * 100
            st.metric(
                f"Percentile among {sex} · {equipment} lifters",
                f"{percentile:.0f}th",
                help=f"Based on {len(peer_group):,} similar lifters in the reference dataset.",
            )
 
    with right:
        # gauge chart against the peer group's typical range
        if len(peer_group) > 20:
            p10, p50, p90 = np.percentile(peer_group["BestDeadliftKg"], [10, 50, 90])
            gauge_max = max(p90 * 1.15, prediction * 1.1)
        else:
            p10, p50, p90 = 0, ref_data["BestDeadliftKg"].median(), ref_data["BestDeadliftKg"].quantile(0.9)
            gauge_max = max(p90 * 1.15, prediction * 1.1)
 
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={"suffix": " kg", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#ff6b35"},
                    "steps": [
                        {"range": [0, p10], "color": "#f3f4f6"},
                        {"range": [p10, p50], "color": "#e5e7eb"},
                        {"range": [p50, p90], "color": "#d1d5db"},
                        {"range": [p90, gauge_max], "color": "#9ca3af"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a1a2e", "width": 3},
                        "thickness": 0.8,
                        "value": p50,
                    },
                },
                title={"text": f"vs {sex} · {equipment} lifters (median marked)"},
            )
        )
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("---")
    st.caption(
        "See the **Visualizations** page to explore the full dataset, or **Model Insights** "
        "for how this model was chosen and how accurate it is."
    )
 
