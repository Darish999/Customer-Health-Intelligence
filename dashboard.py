"""
dashboard.py
------------
Streamlit dashboard for the Customer Health Intelligence System.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Health Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #21262d; }
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 800; color: #00d4aa; }
  .metric-label { font-size: 0.75rem; color: #7d8590; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }
  .risk-critical { color: #ff6b6b; font-weight: 700; }
  .risk-high     { color: #f7a74a; font-weight: 700; }
  .risk-moderate { color: #febc2e; font-weight: 700; }
  .risk-low      { color: #00d4aa; font-weight: 700; }
  h1, h2, h3    { color: #e6edf3 !important; }
  .stMarkdown p { color: #7d8590; }
  .insight-box {
    background: #161b22;
    border-left: 3px solid #00d4aa;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #c9d1d9;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "Critical"      : "#ff6b6b",
    "High Risk"     : "#f7a74a",
    "Moderate Risk" : "#febc2e",
    "Low Risk"      : "#00d4aa",
}
SEGMENT_COLORS = {
    "Retail"  : "#7c6af7",
    "Premium" : "#00d4aa",
    "SME"     : "#f7a74a",
    "HNI"     : "#ff6b6b",
}


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating and scoring data...")
def load_data():
    from data_generator import generate_all
    from ml_engine import run_scoring

    customers, transactions = generate_all()
    scores, full_df = run_scoring(customers, transactions)
    return full_df, transactions


df, transactions = load_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 CHI System")
    st.markdown("*Customer Health Intelligence*")
    st.divider()

    selected_segments = st.multiselect(
        "Customer Segment",
        options=df["segment"].unique().tolist(),
        default=df["segment"].unique().tolist(),
    )
    selected_risk = st.multiselect(
        "Risk Tier",
        options=["Critical", "High Risk", "Moderate Risk", "Low Risk"],
        default=["Critical", "High Risk", "Moderate Risk", "Low Risk"],
    )
    min_health, max_health = st.slider(
        "Health Score Range", 0, 100, (0, 100)
    )
    st.divider()
    show_ai = st.toggle("Enable AI Insights", value=False,
                        help="Requires ANTHROPIC_API_KEY environment variable")
    st.markdown("*Built by Darsh Jogani*")
    st.markdown("[GitHub](https://github.com/Darish999) · [LinkedIn](https://www.linkedin.com/in/darsh-jogani-37b97218b)")

# ── Filter Data ───────────────────────────────────────────────────────────────
filtered = df[
    df["segment"].isin(selected_segments) &
    df["risk_tier"].isin(selected_risk) &
    df["health_score"].between(min_health, max_health)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Customer Health Intelligence System")
st.markdown(f"*Showing {len(filtered):,} of {len(df):,} customers · Real-time churn & health scoring*")
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{len(filtered):,}</div>
        <div class='metric-label'>Total Customers</div>
    </div>""", unsafe_allow_html=True)

with k2:
    avg_health = round(filtered["health_score"].mean(), 1)
    color = "#00d4aa" if avg_health >= 60 else "#f7a74a" if avg_health >= 40 else "#ff6b6b"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:{color}'>{avg_health}</div>
        <div class='metric-label'>Avg Health Score</div>
    </div>""", unsafe_allow_html=True)

with k3:
    critical = int((filtered["risk_tier"] == "Critical").sum())
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#ff6b6b'>{critical:,}</div>
        <div class='metric-label'>Critical Risk</div>
    </div>""", unsafe_allow_html=True)

with k4:
    churned = int(filtered["is_churned"].sum())
    churn_pct = round(churned / max(len(filtered), 1) * 100, 1)
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#f7a74a'>{churn_pct}%</div>
        <div class='metric-label'>Churn Rate</div>
    </div>""", unsafe_allow_html=True)

with k5:
    avg_cs = round(filtered["credit_score"].mean(), 0)
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{int(avg_cs)}</div>
        <div class='metric-label'>Avg Credit Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Risk Tier Distribution")
    risk_counts = filtered["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["risk_tier", "count"]
    fig = px.bar(
        risk_counts, x="risk_tier", y="count",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        showlegend=False, margin=dict(t=10, b=10),
        font_color="#7d8590",
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Health Score vs Churn Score")
    sample = filtered.sample(min(500, len(filtered)), random_state=42)
    fig = px.scatter(
        sample, x="health_score", y="churn_score",
        color="risk_tier", color_discrete_map=RISK_COLORS,
        hover_data=["customer_id", "segment", "credit_score"],
        opacity=0.7, template="plotly_dark",
        labels={"health_score": "Health Score", "churn_score": "Churn Probability"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        margin=dict(t=10, b=10), font_color="#7d8590",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Charts Row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown("### Segment Health Comparison")
    seg_agg = filtered.groupby("segment").agg(
        avg_health=("health_score", "mean"),
        avg_churn=("churn_score", "mean"),
        count=("customer_id", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Health Score", x=seg_agg["segment"], y=seg_agg["avg_health"],
        marker_color="#00d4aa", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        name="Avg Churn Score", x=seg_agg["segment"], y=seg_agg["avg_churn"] * 100,
        mode="lines+markers", marker_color="#f7a74a", yaxis="y2"
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        yaxis=dict(title="Health Score", titlefont_color="#00d4aa"),
        yaxis2=dict(title="Churn % ×100", titlefont_color="#f7a74a",
                    overlaying="y", side="right"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10), font_color="#7d8590",
    )
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.markdown("### Credit Score Distribution by Segment")
    fig = px.violin(
        filtered, x="segment", y="credit_score",
        color="segment", color_discrete_map=SEGMENT_COLORS,
        box=True, template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        showlegend=False, margin=dict(t=10, b=10), font_color="#7d8590",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── AI Insights Panel ─────────────────────────────────────────────────────────
st.markdown("### AI Insights Engine")

if show_ai:
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY environment variable to enable AI insights.")
    else:
        from ai_insights import generate_portfolio_summary, generate_segment_insight, generate_risk_narrative

        tab1, tab2, tab3 = st.tabs(["Portfolio Summary", "Segment Insights", "Risk Tier Analysis"])

        with tab1:
            with st.spinner("Generating portfolio summary..."):
                summary = generate_portfolio_summary(filtered)
            st.markdown(f"<div class='insight-box'>{summary}</div>", unsafe_allow_html=True)

        with tab2:
            seg_choice = st.selectbox("Select Segment", filtered["segment"].unique())
            if st.button("Generate Segment Insight"):
                with st.spinner("Analyzing segment..."):
                    insight = generate_segment_insight(filtered, seg_choice)
                st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

        with tab3:
            tier_choice = st.selectbox("Select Risk Tier", ["Critical", "High Risk", "Moderate Risk", "Low Risk"])
            if st.button("Generate Risk Narrative"):
                with st.spinner("Analyzing risk cohort..."):
                    narrative = generate_risk_narrative(filtered, tier_choice)
                st.markdown(f"<div class='insight-box'>{narrative}</div>", unsafe_allow_html=True)
else:
    st.info("Toggle 'Enable AI Insights' in the sidebar to activate LLM-powered analysis (requires Anthropic API key).")

st.divider()

# ── Customer Table ────────────────────────────────────────────────────────────
st.markdown("### Customer Risk Register")
display_cols = ["customer_id", "segment", "city", "age", "tenure_months",
                "credit_score", "health_score", "churn_score", "risk_tier", "num_products"]

top_risk = filtered.sort_values("churn_score", ascending=False)[display_cols].head(100)
st.dataframe(
    top_risk.style.background_gradient(subset=["churn_score"], cmap="RdYlGn_r")
                  .background_gradient(subset=["health_score"], cmap="RdYlGn"),
    use_container_width=True,
    height=400,
)

# ── Individual Customer Drill-Down ────────────────────────────────────────────
st.divider()
st.markdown("### Customer Drill-Down")
cust_id = st.text_input("Enter Customer ID (e.g. CUST00001):", "CUST00001")

if cust_id:
    row = df[df["customer_id"] == cust_id.upper()]
    if not row.empty:
        row = row.iloc[0]
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Health Score", row["health_score"])
            st.metric("Credit Score", row["credit_score"])
        with d2:
            st.metric("Churn Score", f"{row['churn_score']:.2%}")
            st.metric("Risk Tier", row["risk_tier"])
        with d3:
            st.metric("Segment", row["segment"])
            st.metric("Tenure", f"{row['tenure_months']} months")

        cust_txns = transactions[transactions["customer_id"] == cust_id.upper()]
        if not cust_txns.empty:
            fig = px.line(
                cust_txns, x="month", y=["avg_balance", "total_credit", "total_debit"],
                template="plotly_dark",
                title=f"Transaction History — {cust_id}",
                color_discrete_sequence=["#00d4aa", "#7c6af7", "#f7a74a"],
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font_color="#7d8590", margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        if show_ai and os.getenv("ANTHROPIC_API_KEY"):
            from ai_insights import generate_customer_insight
            if st.button("Generate RM Briefing Note"):
                with st.spinner("Generating customer briefing..."):
                    note = generate_customer_insight(row, cust_txns)
                st.markdown(f"<div class='insight-box'>{note}</div>", unsafe_allow_html=True)
    else:
        st.error(f"Customer {cust_id} not found.")
