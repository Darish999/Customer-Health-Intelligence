"""
etl_pipeline.py
---------------
Loads generated customer and transaction data into MySQL.
Includes auto-schema creation, chunked inserts, and retry logic.
"""

import time
import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error
from data_generator import generate_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",
    "password": "your_password",
    "database": "customer_health_db",
}
CHUNK_SIZE = 200
MAX_RETRIES = 3


# ── Schema ───────────────────────────────────────────────────────────────────
SCHEMAS = {
    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id       VARCHAR(12) PRIMARY KEY,
            name              VARCHAR(100),
            age               INT,
            city              VARCHAR(50),
            occupation        VARCHAR(50),
            segment           VARCHAR(20),
            tenure_months     INT,
            monthly_income    DECIMAL(14,2),
            num_products      INT,
            credit_score      INT,
            loan_outstanding  DECIMAL(14,2),
            churn_probability DECIMAL(6,4),
            is_churned        TINYINT(1),
            onboard_date      DATE,
            products_held     TEXT,
            health_score      DECIMAL(5,2) DEFAULT NULL,
            risk_tier         VARCHAR(20)  DEFAULT NULL,
            updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """,
    "transactions": """
        CREATE TABLE IF NOT EXISTS transactions (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            customer_id   VARCHAR(12),
            month         VARCHAR(7),
            txn_count     INT,
            total_debit   DECIMAL(14,2),
            total_credit  DECIMAL(14,2),
            avg_balance   DECIMAL(14,2),
            digital_txns  INT,
            complaints    INT,
            login_count   INT,
            UNIQUE KEY uq_cust_month (customer_id, month),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """,
    "ml_scores": """
        CREATE TABLE IF NOT EXISTS ml_scores (
            customer_id       VARCHAR(12) PRIMARY KEY,
            churn_score       DECIMAL(6,4),
            health_score      DECIMAL(5,2),
            risk_tier         VARCHAR(20),
            segment_cluster   INT,
            scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """,
}


def get_connection(retries: int = MAX_RETRIES):
    for attempt in range(1, retries + 1):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            log.info("DB connection established")
            return conn
        except Error as e:
            log.warning(f"Connection attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError("Could not connect to MySQL after retries")


def ensure_tables(conn):
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")
    for table, ddl in SCHEMAS.items():
        cursor.execute(ddl)
        log.info(f"Table ensured: {table}")
    conn.commit()
    cursor.close()


def chunked_upsert(conn, df: pd.DataFrame, table: str, pk_col: str):
    cursor = conn.cursor()
    cols   = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join([f"{c}=VALUES({c})" for c in cols if c != pk_col])
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )
    total = 0
    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i : i + CHUNK_SIZE]
        rows  = [tuple(r) for r in chunk.itertuples(index=False)]
        cursor.executemany(sql, rows)
        conn.commit()
        total += len(chunk)
        log.info(f"  [{table}] Inserted/updated {total}/{len(df)} rows")
    cursor.close()


def run():
    customers, transactions = generate_all()

    conn = get_connection()
    ensure_tables(conn)

    log.info("Loading customers...")
    chunked_upsert(conn, customers, "customers", "customer_id")

    log.info("Loading transactions...")
    chunked_upsert(conn, transactions, "transactions", "id")

    conn.close()
    log.info("ETL complete.")


if __name__ == "__main__":
    run()
