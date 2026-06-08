# Customer Health Intelligence System 🏦

> End-to-end banking customer analytics — churn prediction, financial health scoring, AI-powered insights, and interactive dashboard.

Built by **[Darsh Jogani](https://www.linkedin.com/in/darsh-jogani-37b97218b)** | MS Business Analytics & AI, UT Dallas | FRM Candidate

---

## Overview

The Customer Health Intelligence (CHI) System is a production-grade analytics platform that simulates the kind of customer intelligence infrastructure used by GCC banking analytics teams. It ingests customer and transaction data, runs ML-based churn and health scoring, and surfaces LLM-generated insights through an interactive Streamlit dashboard.

This project directly mirrors real analytical work done in banking contexts — including the treasury management system implementations at Credence Analytics and the ETL pipeline architecture built at Jindal Pipe USA.

---

## Architecture

```
┌─────────────────────┐
│  data_generator.py  │  Synthetic bank customers + transaction histories
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   etl_pipeline.py   │  Auto-schema MySQL · Chunked inserts · Retry logic
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    ml_engine.py     │  GBM Churn Model · Health Score · KMeans Clustering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ai_insights.py    │  Claude API · Portfolio · Segment · Customer briefs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    dashboard.py     │  Streamlit · Plotly · Real-time filters · Drill-down
└─────────────────────┘
```

---

## Features

### Data Layer
- **1,000 synthetic bank customers** across Retail, Premium, SME, and HNI segments
- **12 months** of transaction history per customer (12,000 records)
- Realistic churn signals: declining transaction activity, complaints, login drops

### ML Scoring Engine
- **Churn Model** — Gradient Boosting Classifier with 19 engineered features
- **Health Score** — Composite 0–100 metric (credit, balance, tenure, engagement)
- **Risk Tiers** — Low Risk / Moderate Risk / High Risk / Critical
- **Customer Segmentation** — KMeans clustering on behavioral features

### AI Insights (Claude API)
- Portfolio-level executive summary for CRO briefing
- Segment-level risk narrative
- Individual customer RM briefing notes
- Risk tier cohort analysis

### Dashboard
- Real-time KPI cards (health, churn, critical count)
- Risk tier distribution bar chart
- Health vs Churn scatter plot
- Segment comparison dual-axis chart
- Credit score violin plots by segment
- Customer risk register with sortable table
- Individual customer drill-down with transaction timeline

---

## Setup

### Prerequisites
- Python 3.11+
- MySQL 8.0+ (optional — dashboard runs without it)
- Anthropic API key (optional — for AI insights)

### Installation

```bash
git clone https://github.com/Darish999/Customer-Health-Intelligence.git
cd Customer-Health-Intelligence

pip install -r requirements.txt
```

### Run Dashboard (no MySQL needed)

```bash
streamlit run dashboard.py
```

### Full Pipeline with MySQL

```bash
# 1. Configure DB credentials in etl_pipeline.py
# 2. Run ETL
python etl_pipeline.py

# 3. Set API key for AI insights
export ANTHROPIC_API_KEY=your_key_here

# 4. Launch dashboard
streamlit run dashboard.py
```

---

## Project Structure

```
customer-health-intelligence/
├── data_generator.py   # Synthetic data generation (Faker, NumPy)
├── etl_pipeline.py     # MySQL ETL with auto-schema, chunked inserts
├── ml_engine.py        # Churn model, health scoring, segmentation
├── ai_insights.py      # Claude API — portfolio, segment, customer insights
├── dashboard.py        # Full Streamlit dashboard with Plotly
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Python, Faker, NumPy, Pandas |
| Storage | MySQL 8.0, mysql-connector-python |
| ML | scikit-learn (GBM, KMeans, StandardScaler) |
| AI Insights | Anthropic Claude API (claude-sonnet) |
| Visualization | Plotly Express, Plotly Graph Objects |
| Dashboard | Streamlit |

---

## Key ML Details

**Churn Model Features (19):**
- Demographics: age, tenure, income, credit score
- Product: num_products, loan_outstanding
- Behavioral: avg_txn_count, avg_balance, login_count, txn_trend
- Derived: income_to_loan_ratio, balance_income_ratio, complaint_rate, digital_ratio

**Health Score Formula:**
```
Health Score = Credit Score Component (0-25)
             + Balance/Income Ratio     (0-20)
             + Product Depth            (0-20)
             + Tenure Loyalty           (0-15)
             + Digital Engagement       (0-10)
             - Complaint Penalty        (0-10)
             + Transaction Trend        (0-10)
```

---

## About the Author

**Darsh Jogani** is a Business & Finance Analyst with deep experience in banking analytics and financial data systems. Currently building end-to-end analytics infrastructure at Jindal Pipe USA (SAP ETL pipelines, Power BI dashboards, full-stack tools), with prior experience implementing treasury management software at Credence Analytics, Mumbai.

- FRM Part 1 Cleared · Part 2 — November 2026
- MS Business Analytics & AI — UT Dallas (May 2026)
- Targeting senior analytics and risk roles at GCCs and Big 4 firms

🔗 [LinkedIn](https://www.linkedin.com/in/darsh-jogani-37b97218b) · [Portfolio](https://darish999.github.io/Darshjogani.github.io/) · [GitHub](https://github.com/Darish999)
