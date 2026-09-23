"""Visualizations page — explore the dataset behind the model."""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .main > div {padding-top: 1.5rem;}
        #MainMenu, footer {visibility: hidden;}
        .app-title {font-size: 1.9rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.1rem;}
        .app-subtitle {font-size: 0.95rem; color: #6b7280; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("lifting_data_sample.csv")


df = load_data()

st.markdown('<div class="app-title">Explore the Data</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="app-subtitle">A {len(df):,}-row sample of cleaned lifting records '
    "used to train the model.</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------
f1, f2 = st.columns(2)
sex_filter = f1.multiselect("Sex", sorted(df["Sex"].unique()), default=sorted(df["Sex"].unique()))
equip_filter = f2.multiselect(
    "Equipment", sorted(df["Equipment"].unique()), default=sorted(df["Equipment"].unique())
)
view = df[df["Sex"].isin(sex_filter) & df["Equipment"].isin(equip_filter)]
st.caption(f"Showing {len(view):,} of {len(df):,} rows based on the filters above.")

# ------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------
st.markdown("#### Distributions")
d1, d2, d3, d4 = st.columns(4)
for col, feature, title in zip(
    [d1, d2, d3, d4],
    ["BestDeadliftKg", "BestSquatKg", "BestBenchKg", "BodyweightKg"],
    ["Deadlift", "Squat", "Bench", "Bodyweight"],
):
    fig = px.histogram(view, x=feature, nbins=40, color_discrete_sequence=["#ff6b35"])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title=title, showlegend=False)
    col.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Relationship with the target
# ------------------------------------------------------------------
st.markdown("#### Squat, Bench and Bodyweight vs Deadlift")
x_axis = st.selectbox("Compare Deadlift against", ["BestSquatKg", "BestBenchKg", "BodyweightKg"])
color_by = st.radio("Color by", ["Sex", "Equipment"], horizontal=True)

scatter = px.scatter(
    view.sample(min(4000, len(view)), random_state=42),
    x=x_axis,
    y="BestDeadliftKg",
    color=color_by,
    opacity=0.45,
    trendline="ols",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
scatter.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(scatter, use_container_width=True)

corr = view[["BestSquatKg", "BestBenchKg", "BodyweightKg", "BestDeadliftKg"]].corr()["BestDeadliftKg"]
st.caption(
    f"Correlation with Deadlift — Squat: **{corr['BestSquatKg']:.2f}**, "
    f"Bench: **{corr['BestBenchKg']:.2f}**, Bodyweight: **{corr['BodyweightKg']:.2f}**."
)

# ------------------------------------------------------------------
# Group comparisons
# ------------------------------------------------------------------
st.markdown("#### Deadlift by Group")
g1, g2 = st.columns(2)
with g1:
    box1 = px.box(view, x="Sex", y="BestDeadliftKg", color="Sex", color_discrete_sequence=px.colors.qualitative.Set2)
    box1.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(box1, use_container_width=True)
with g2:
    box2 = px.box(
        view, x="Equipment", y="BestDeadliftKg", color="Equipment",
        category_orders={"Equipment": ["Raw", "Wraps", "Single-ply", "Multi-ply"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    box2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(box2, use_container_width=True)

# ------------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------------
st.markdown("#### Correlation Matrix")
heat_df = view[["BestSquatKg", "BestBenchKg", "BodyweightKg", "BestDeadliftKg"]].corr()
heat = px.imshow(heat_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
heat.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(heat, use_container_width=True)
