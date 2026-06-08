"""
data_generator.py
-----------------
Generates synthetic bank customer profiles and transaction histories
for the Customer Health Intelligence System.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CUSTOMERS   = 1000
NUM_MONTHS      = 12
START_DATE      = datetime(2024, 1, 1)

SEGMENTS        = ["Retail", "Premium", "SME", "HNI"]
PRODUCTS        = ["Savings", "Current", "FD", "Loan", "Credit Card", "Mutual Fund", "Insurance"]
CITIES          = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
OCCUPATIONS     = ["Salaried", "Self-Employed", "Business Owner", "Retired", "Student"]


def generate_customers(n: int = NUM_CUSTOMERS) -> pd.DataFrame:
    """Generate synthetic customer master data."""
    records = []
    for i in range(1, n + 1):
        segment    = random.choices(SEGMENTS, weights=[55, 25, 12, 8])[0]
        occupation = random.choices(OCCUPATIONS, weights=[45, 20, 20, 10, 5])[0]

        # Income varies by segment
        income_map = {"Retail": (25000, 80000), "Premium": (80000, 250000),
                      "SME": (100000, 500000), "HNI": (300000, 2000000)}
        lo, hi = income_map[segment]
        monthly_income = round(random.uniform(lo, hi), -2)

        age = int(np.clip(np.random.normal(38, 10), 21, 70))
        tenure_months = int(np.clip(np.random.exponential(36), 1, 240))
        num_products = random.choices([1, 2, 3, 4, 5], weights=[30, 30, 20, 12, 8])[0]

        # Churn probability influenced by tenure, products, income
        churn_base = 0.15
        if tenure_months < 6:  churn_base += 0.20
        if num_products == 1:  churn_base += 0.15
        if segment == "HNI":   churn_base -= 0.08
        churn_prob = float(np.clip(churn_base + np.random.normal(0, 0.05), 0.02, 0.85))

        records.append({
            "customer_id"     : f"CUST{i:05d}",
            "name"            : fake.name(),
            "age"             : age,
            "city"            : random.choice(CITIES),
            "occupation"      : occupation,
            "segment"         : segment,
            "tenure_months"   : tenure_months,
            "monthly_income"  : monthly_income,
            "num_products"    : num_products,
            "credit_score"    : int(np.clip(np.random.normal(700, 80), 300, 900)),
            "loan_outstanding": round(random.uniform(0, monthly_income * 24), -3) if random.random() < 0.45 else 0,
            "churn_probability": round(churn_prob, 4),
            "is_churned"      : int(random.random() < churn_prob * 0.4),
            "onboard_date"    : (datetime.today() - timedelta(days=tenure_months * 30)).strftime("%Y-%m-%d"),
            "products_held"   : ", ".join(random.sample(PRODUCTS, num_products)),
        })

    return pd.DataFrame(records)


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly transaction summaries per customer."""
    rows = []
    for _, cust in customers.iterrows():
        base_spend = cust["monthly_income"] * random.uniform(0.3, 0.9)
        for m in range(NUM_MONTHS):
            month_dt = START_DATE + timedelta(days=m * 30)
            # Declining activity signal for churned customers in later months
            activity_factor = 1.0
            if cust["is_churned"] and m > 8:
                activity_factor = random.uniform(0.1, 0.5)

            txn_count  = max(1, int(np.random.poisson(18) * activity_factor))
            debit_amt  = round(base_spend * activity_factor * random.uniform(0.8, 1.2), 2)
            credit_amt = round(cust["monthly_income"] * random.uniform(0.9, 1.1), 2)
            avg_bal    = round(max(0, credit_amt - debit_amt + random.uniform(-5000, 20000)), 2)

            rows.append({
                "customer_id"   : cust["customer_id"],
                "month"         : month_dt.strftime("%Y-%m"),
                "txn_count"     : txn_count,
                "total_debit"   : debit_amt,
                "total_credit"  : credit_amt,
                "avg_balance"   : avg_bal,
                "digital_txns"  : int(txn_count * random.uniform(0.4, 0.95)),
                "complaints"    : int(np.random.poisson(0.1 * (1 + cust["churn_probability"]))),
                "login_count"   : max(0, int(np.random.poisson(12) * activity_factor)),
            })

    return pd.DataFrame(rows)


def generate_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Generating customers...")
    customers = generate_customers()
    print(f"  → {len(customers)} customers created")

    print("Generating transactions...")
    transactions = generate_transactions(customers)
    print(f"  → {len(transactions)} monthly transaction records created")

    return customers, transactions


if __name__ == "__main__":
    customers, transactions = generate_all()
    customers.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)
    print("Saved to data/")
