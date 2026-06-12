"""
generate_data.py
----------------
Generates a realistic synthetic telecom churn dataset.
Run standalone to produce churn_data.csv, or import generate_dataset()
directly from the pipeline.

Churn drivers modelled:
  - Short tenure → higher churn
  - High monthly charges → higher churn
  - Month-to-month contract → much higher churn
  - Many support calls → higher churn
  - No tech support / online security → higher churn
  - Senior citizens → slightly higher churn
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


def generate_dataset(n_customers: int = 5000) -> pd.DataFrame:
    n = n_customers

    # ── Demographics ──────────────────────────────────────────────────────────
    senior_citizen   = RNG.binomial(1, 0.16, n)
    partner          = RNG.binomial(1, 0.48, n)
    dependents       = RNG.binomial(1, 0.30, n)

    # ── Account info ──────────────────────────────────────────────────────────
    tenure_months    = RNG.integers(1, 73, n)           # 1–72 months
    # Contract: 0=Month-to-month, 1=One year, 2=Two year
    contract         = RNG.choice([0, 1, 2], n, p=[0.55, 0.24, 0.21])
    paperless_billing = RNG.binomial(1, 0.59, n)
    # Payment method: 0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card
    payment_method   = RNG.choice([0, 1, 2, 3], n, p=[0.34, 0.23, 0.22, 0.21])

    # ── Services ──────────────────────────────────────────────────────────────
    phone_service    = RNG.binomial(1, 0.90, n)
    multiple_lines   = RNG.binomial(1, 0.42, n) * phone_service
    internet_service = RNG.choice([0, 1, 2], n, p=[0.22, 0.34, 0.44])   # 0=None,1=DSL,2=Fiber
    online_security  = RNG.binomial(1, 0.38, n) * (internet_service > 0)
    online_backup    = RNG.binomial(1, 0.44, n) * (internet_service > 0)
    device_protection = RNG.binomial(1, 0.44, n) * (internet_service > 0)
    tech_support     = RNG.binomial(1, 0.33, n) * (internet_service > 0)
    streaming_tv     = RNG.binomial(1, 0.38, n) * (internet_service > 0)
    streaming_movies = RNG.binomial(1, 0.39, n) * (internet_service > 0)

    # ── Charges ───────────────────────────────────────────────────────────────
    base_charge = 18.0
    monthly_charges = (
        base_charge
        + phone_service      * RNG.uniform(10, 30, n)
        + multiple_lines     * RNG.uniform(5,  20, n)
        + (internet_service == 1) * RNG.uniform(20, 40, n)
        + (internet_service == 2) * RNG.uniform(50, 80, n)
        + online_security    * RNG.uniform(5, 15, n)
        + online_backup      * RNG.uniform(5, 15, n)
        + device_protection  * RNG.uniform(5, 15, n)
        + tech_support       * RNG.uniform(5, 15, n)
        + streaming_tv       * RNG.uniform(5, 15, n)
        + streaming_movies   * RNG.uniform(5, 15, n)
    ).round(2)

    total_charges = (monthly_charges * tenure_months * RNG.uniform(0.95, 1.05, n)).round(2)

    # ── Support interactions ──────────────────────────────────────────────────
    support_calls = RNG.poisson(lam=1.5 + (internet_service == 2) * 0.8, size=n)
    support_calls = np.clip(support_calls, 0, 10)

    # ── Churn label (logistic model) ─────────────────────────────────────────
    logit = (
        -3.5
        + 0.04  * (72 - tenure_months)            # newer customers churn more
        + 0.015 * (monthly_charges - 60)          # high charges increase churn
        - 1.20  * (contract == 1)                 # 1-yr contract reduces churn
        - 2.10  * (contract == 2)                 # 2-yr contract reduces churn even more
        + 0.25  * (internet_service == 2)         # fiber optic users churn more
        - 0.40  * online_security                 # security service retains customers
        - 0.35  * tech_support                    # tech support retains customers
        + 0.30  * support_calls                   # more support calls → more frustration
        + 0.18  * senior_citizen                  # seniors churn slightly more
        + 0.12  * (payment_method == 0)           # electronic check → slightly higher churn
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, prob_churn, n)

    df = pd.DataFrame({
        "customer_id":       [f"CUST-{i:05d}" for i in range(n)],
        "senior_citizen":    senior_citizen,
        "partner":           partner,
        "dependents":        dependents,
        "tenure_months":     tenure_months,
        "phone_service":     phone_service,
        "multiple_lines":    multiple_lines,
        "internet_service":  internet_service,
        "online_security":   online_security,
        "online_backup":     online_backup,
        "device_protection": device_protection,
        "tech_support":      tech_support,
        "streaming_tv":      streaming_tv,
        "streaming_movies":  streaming_movies,
        "contract":          contract,
        "paperless_billing": paperless_billing,
        "payment_method":    payment_method,
        "monthly_charges":   monthly_charges,
        "total_charges":     total_charges,
        "support_calls":     support_calls.astype(int),
        "churn":             churn,
    })

    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "churn_data.csv"
    df  = generate_dataset(5000)
    df.to_csv(out, index=False)
    churn_rate = df["churn"].mean() * 100
    print(f"Generated {len(df):,} rows → {out}")
    print(f"Churn rate: {churn_rate:.1f}%")
    print(df.head())
