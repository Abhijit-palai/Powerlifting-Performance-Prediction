"""Model Insights page — how the model was chosen and how good it is."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# pages/ is one level below the main app folder, where the data files live
APP_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
        .main > div {padding-top: 1.5rem;}
        #MainMenu, footer {visibility: hidden;}
        .app-title {font-size: 1.9rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.1rem;}
        .app-subtitle {font-size: 0.95rem; color: #6b7280; margin-bottom: 1.2rem;}
        div[data-testid="stMetric"] {
            background: #f8f9fb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.8rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    with open(APP_DIR / "metrics.json") as f:
        metrics = json.load(f)
    comparison = pd.read_csv(APP_DIR / "model_comparison.csv")
    importance = pd.read_csv(APP_DIR / "feature_importance.csv")
    actual_vs_pred = pd.read_csv(APP_DIR / "actual_vs_predicted_sample.csv")
    return metrics, comparison, importance, actual_vs_pred


metrics, comparison_df, importance_df, av_df = load_artifacts()

st.markdown('<div class="app-title">Model Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">How the final model was chosen, and how accurate it is on '
    "data it never saw during training.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Headline metrics
# ------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{metrics['r2']:.4f}")
m2.metric("Adjusted R²", f"{metrics['adjusted_r2']:.4f}")
m3.metric("MAE", f"{metrics['mae']:.2f} kg")
m4.metric("RMSE", f"{metrics['rmse']:.2f} kg")
st.caption(
    f"Evaluated on {metrics['n_test']:,} held-out test rows "
    f"(trained on {metrics['n_train']:,} rows). Final model: **tuned XGBoost**."
)

st.info(
    "R² is high here (~0.88) mainly because squat and bench are strong predictors of "
    "deadlift for the same lifter on the same day — this model answers *'given today's "
    "squat and bench, what will the deadlift be?'*, which is an easier question than "
    "predicting strength from demographics alone.",
    icon="ℹ️",
)

st.markdown("---")

# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------
st.markdown("#### Model Comparison")
st.caption("Every regression model tested, before hyperparameter tuning, sorted by Test R².")

comparison_df = comparison_df.sort_values("Test R2", ascending=True)
colors = ["#ff6b35" if m == "XGBoost" else "#9ca3af" for m in comparison_df["Model"]]
bar = go.Figure(
    go.Bar(
        x=comparison_df["Test R2"],
        y=comparison_df["Model"],
        orientation="h",
        marker_color=colors,
        text=comparison_df["Test R2"].round(3),
        textposition="outside",
    )
)
bar.update_layout(height=430, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Test R²")
st.plotly_chart(bar, use_container_width=True)

with st.expander("See full comparison table"):
    st.dataframe(comparison_df.sort_values("Test R2", ascending=False), use_container_width=True, hide_index=True)

st.markdown("---")

# ------------------------------------------------------------------
# Feature importance + Actual vs Predicted
# ------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Feature Importance")
    imp = importance_df.sort_values("importance", ascending=True)
    imp_fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=["#1a1a2e"])
    imp_fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(imp_fig, use_container_width=True)
    st.caption("Squat dominates, followed by Sex and Bench — consistent with the correlations on the Visualizations page.")

with c2:
    st.markdown("#### Actual vs Predicted")
    scatter = px.scatter(av_df, x="actual", y="predicted", opacity=0.35, color_discrete_sequence=["#ff6b35"])
    max_val = max(av_df["actual"].max(), av_df["predicted"].max())
    scatter.add_trace(
        go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(color="#1a1a2e", dash="dash"), name="Perfect prediction")
    )
    scatter.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Actual (kg)", yaxis_title="Predicted (kg)", showlegend=False)
    st.plotly_chart(scatter, use_container_width=True)
    st.caption("Points close to the dashed line are accurate predictions; the spread shows typical error.")
    