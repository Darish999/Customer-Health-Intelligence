"""
ml_engine.py
------------
Churn scoring, financial health scoring, and customer segmentation
for the Customer Health Intelligence System.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# ── Feature Engineering ──────────────────────────────────────────────────────
def build_features(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Merge and engineer features from customer + transaction data."""

    # Aggregate transaction stats per customer
    txn_agg = transactions.groupby("customer_id").agg(
        avg_monthly_txns    = ("txn_count",    "mean"),
        avg_balance         = ("avg_balance",  "mean"),
        avg_debit           = ("total_debit",  "mean"),
        avg_credit          = ("total_credit", "mean"),
        total_complaints    = ("complaints",   "sum"),
        avg_login           = ("login_count",  "mean"),
        avg_digital_ratio   = ("digital_txns", "mean"),
        balance_std         = ("avg_balance",  "std"),
        txn_trend           = ("txn_count",    lambda x: float(np.polyfit(range(len(x)), x, 1)[0]))
    ).reset_index()

    df = customers.merge(txn_agg, on="customer_id", how="left")

    # Derived features
    df["income_to_loan_ratio"]   = df["monthly_income"] / (df["loan_outstanding"] + 1)
    df["balance_income_ratio"]   = df["avg_balance"] / (df["monthly_income"] + 1)
    df["digital_ratio"]          = df["avg_digital_ratio"] / (df["avg_monthly_txns"] + 1)
    df["complaint_rate"]         = df["total_complaints"] / (df["tenure_months"] + 1)
    df["segment_enc"]            = df["segment"].map({"Retail": 0, "Premium": 1, "SME": 2, "HNI": 3})

    df.fillna(0, inplace=True)
    return df


FEATURE_COLS = [
    "age", "tenure_months", "monthly_income", "num_products",
    "credit_score", "loan_outstanding", "avg_monthly_txns",
    "avg_balance", "avg_debit", "avg_credit", "total_complaints",
    "avg_login", "balance_std", "txn_trend", "income_to_loan_ratio",
    "balance_income_ratio", "digital_ratio", "complaint_rate", "segment_enc",
]


# ── Churn Model ──────────────────────────────────────────────────────────────
def train_churn_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["is_churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print("\n── Churn Model Performance ──")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return model, scaler


def score_churn(df: pd.DataFrame, model, scaler) -> pd.Series:
    X_s = scaler.transform(df[FEATURE_COLS])
    return pd.Series(model.predict_proba(X_s)[:, 1], index=df.index)


# ── Health Score ─────────────────────────────────────────────────────────────
def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite financial health score (0–100).
    Higher = healthier customer relationship.
    """
    s = pd.Series(0.0, index=df.index)

    # Credit score component (0-25)
    s += np.clip((df["credit_score"] - 300) / 600 * 25, 0, 25)

    # Balance-to-income component (0-20)
    bri = np.clip(df["balance_income_ratio"], 0, 3)
    s += bri / 3 * 20

    # Product depth (0-20)
    s += np.clip(df["num_products"] / 5 * 20, 0, 20)

    # Tenure loyalty (0-15)
    s += np.clip(df["tenure_months"] / 120 * 15, 0, 15)

    # Digital engagement (0-10)
    s += np.clip(df["digital_ratio"] * 10, 0, 10)

    # Complaint penalty (0 to -10)
    s -= np.clip(df["complaint_rate"] * 20, 0, 10)

    # Txn activity trend (0-10)
    trend_norm = np.clip(df["txn_trend"] + 5, 0, 10)
    s += trend_norm

    return np.clip(s, 0, 100).round(2)


# ── Segmentation ─────────────────────────────────────────────────────────────
def cluster_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.Series:
    cluster_features = ["monthly_income", "avg_balance", "num_products",
                        "credit_score", "avg_monthly_txns"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cluster_features].fillna(0))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return pd.Series(km.fit_predict(X), index=df.index)


# ── Risk Tier ────────────────────────────────────────────────────────────────
def assign_risk_tier(churn_score: pd.Series) -> pd.Series:
    return pd.cut(
        churn_score,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical"],
    ).astype(str)


# ── Master Score Function ────────────────────────────────────────────────────
def run_scoring(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    print("Building features...")
    df = build_features(customers, transactions)

    print("Training churn model...")
    model, scaler = train_churn_model(df)

    print("Scoring all customers...")
    df["churn_score"]     = score_churn(df, model, scaler).round(4)
    df["health_score"]    = compute_health_score(df)
    df["risk_tier"]       = assign_risk_tier(df["churn_score"])
    df["segment_cluster"] = cluster_customers(df)

    scores = df[["customer_id", "churn_score", "health_score", "risk_tier", "segment_cluster"]]
    print(f"\nScoring complete. Risk distribution:\n{df['risk_tier'].value_counts()}")
    return scores, df


if __name__ == "__main__":
    from data_generator import generate_all
    customers, transactions = generate_all()
    scores, full_df = run_scoring(customers, transactions)
    scores.to_csv("data/scores.csv", index=False)
    print("Scores saved to data/scores.csv")
