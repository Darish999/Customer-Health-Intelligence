"""
ai_insights.py
--------------
LLM-powered natural language insight generation for customer segments,
risk tiers, and individual customer profiles using the Anthropic API.
"""

import os
import json
import anthropic
import pandas as pd

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "your-key-here"))
MODEL  = "claude-sonnet-4-20250514"


# ── Portfolio-Level Insights ─────────────────────────────────────────────────
def generate_portfolio_summary(df: pd.DataFrame) -> str:
    """Generate an executive summary of the entire customer portfolio."""

    stats = {
        "total_customers"     : int(len(df)),
        "avg_health_score"    : round(float(df["health_score"].mean()), 1),
        "avg_churn_score"     : round(float(df["churn_score"].mean()), 3),
        "critical_customers"  : int((df["risk_tier"] == "Critical").sum()),
        "high_risk_customers" : int((df["risk_tier"] == "High Risk").sum()),
        "churned_count"       : int(df["is_churned"].sum()),
        "avg_credit_score"    : round(float(df["credit_score"].mean()), 0),
        "segment_breakdown"   : df["segment"].value_counts().to_dict(),
        "avg_products"        : round(float(df["num_products"].mean()), 1),
        "total_loan_exposure" : round(float(df["loan_outstanding"].sum()), 0),
    }

    prompt = f"""
You are a senior banking analytics consultant preparing a briefing for the Chief Risk Officer.

Here is the current customer portfolio snapshot:
{json.dumps(stats, indent=2)}

Generate a concise executive insight report (5–7 sentences) covering:
1. Overall portfolio health assessment
2. Key churn risk alert with specific numbers
3. Segment-level observation
4. One actionable recommendation for the risk team

Be direct, data-driven, and use banking terminology. No bullet points — write in clear paragraphs.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Segment-Level Insights ───────────────────────────────────────────────────
def generate_segment_insight(df: pd.DataFrame, segment: str) -> str:
    """Generate targeted insight for a specific customer segment."""

    seg_df = df[df["segment"] == segment]
    if seg_df.empty:
        return f"No data available for segment: {segment}"

    stats = {
        "segment"           : segment,
        "count"             : int(len(seg_df)),
        "avg_health_score"  : round(float(seg_df["health_score"].mean()), 1),
        "avg_churn_score"   : round(float(seg_df["churn_score"].mean()), 3),
        "critical_count"    : int((seg_df["risk_tier"] == "Critical").sum()),
        "avg_income"        : round(float(seg_df["monthly_income"].mean()), 0),
        "avg_tenure_months" : round(float(seg_df["tenure_months"].mean()), 0),
        "avg_products"      : round(float(seg_df["num_products"].mean()), 1),
        "avg_credit_score"  : round(float(seg_df["credit_score"].mean()), 0),
    }

    prompt = f"""
You are a banking customer analytics specialist.

Analyze this customer segment data and generate a 3–4 sentence insight:
{json.dumps(stats, indent=2)}

Cover: health and churn risk profile, what's driving risk, and one targeted retention action.
Be specific and use the numbers provided.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Individual Customer Insights ─────────────────────────────────────────────
def generate_customer_insight(customer_row: pd.Series, recent_txns: pd.DataFrame) -> str:
    """Generate a relationship manager briefing for a single customer."""

    txn_summary = {
        "avg_monthly_balance": round(float(recent_txns["avg_balance"].mean()), 0),
        "avg_txn_count"      : round(float(recent_txns["txn_count"].mean()), 0),
        "total_complaints"   : int(recent_txns["complaints"].sum()),
        "avg_login_count"    : round(float(recent_txns["login_count"].mean()), 0),
        "months_observed"    : int(len(recent_txns)),
    }

    customer_info = {
        "customer_id"      : customer_row["customer_id"],
        "segment"          : customer_row["segment"],
        "age"              : int(customer_row["age"]),
        "tenure_months"    : int(customer_row["tenure_months"]),
        "products_held"    : customer_row["products_held"],
        "churn_score"      : float(customer_row["churn_score"]),
        "health_score"     : float(customer_row["health_score"]),
        "risk_tier"        : customer_row["risk_tier"],
        "credit_score"     : int(customer_row["credit_score"]),
        "monthly_income"   : float(customer_row["monthly_income"]),
        "loan_outstanding" : float(customer_row["loan_outstanding"]),
    }

    prompt = f"""
You are a relationship manager assistant at a bank. Generate a concise customer briefing note.

Customer Profile:
{json.dumps(customer_info, indent=2)}

Recent Transaction Summary:
{json.dumps(txn_summary, indent=2)}

Write a 3-sentence briefing covering:
1. Current relationship health and risk status
2. Key signals from transaction behavior
3. Recommended next action for the relationship manager

Be specific, professional, and action-oriented.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# ── Risk Tier Narrative ──────────────────────────────────────────────────────
def generate_risk_narrative(df: pd.DataFrame, risk_tier: str) -> str:
    """Generate a narrative for a risk tier cohort."""

    tier_df = df[df["risk_tier"] == risk_tier]
    if tier_df.empty:
        return f"No customers in {risk_tier} tier."

    stats = {
        "risk_tier"          : risk_tier,
        "count"              : int(len(tier_df)),
        "pct_of_portfolio"   : round(len(tier_df) / len(df) * 100, 1),
        "avg_health_score"   : round(float(tier_df["health_score"].mean()), 1),
        "avg_tenure"         : round(float(tier_df["tenure_months"].mean()), 0),
        "top_segments"       : tier_df["segment"].value_counts().head(2).to_dict(),
        "avg_products"       : round(float(tier_df["num_products"].mean()), 1),
        "avg_credit_score"   : round(float(tier_df["credit_score"].mean()), 0),
    }

    prompt = f"""
You are a bank's Chief Risk Officer reviewing a customer risk cohort.

Risk Tier Data:
{json.dumps(stats, indent=2)}

In 3 sentences, describe: the nature of this cohort's risk profile, 
what common characteristics drive them into this tier, 
and the priority intervention strategy.

Be precise and actionable.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


if __name__ == "__main__":
    from data_generator import generate_all
    from ml_engine import run_scoring

    customers, transactions = generate_all()
    _, scored_df = run_scoring(customers, transactions)

    print("\n── Portfolio Summary ──")
    print(generate_portfolio_summary(scored_df))

    print("\n── HNI Segment Insight ──")
    print(generate_segment_insight(scored_df, "HNI"))

    print("\n── Critical Risk Tier Narrative ──")
    print(generate_risk_narrative(scored_df, "Critical"))
