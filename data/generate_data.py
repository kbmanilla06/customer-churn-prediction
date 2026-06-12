import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


def generate_dataset(n_customers: int = 5000) -> pd.DataFrame:
    n = n_customers

    senior_citizen    = RNG.binomial(1, 0.16, n)
    partner           = RNG.binomial(1, 0.48, n)
    dependents        = RNG.binomial(1, 0.30, n)

    tenure_months     = RNG.integers(1, 73, n)
    contract          = RNG.choice([0, 1, 2], n, p=[0.55, 0.24, 0.21])
    paperless_billing = RNG.binomial(1, 0.59, n)
    payment_method    = RNG.choice([0, 1, 2, 3], n, p=[0.34, 0.23, 0.22, 0.21])

    phone_service     = RNG.binomial(1, 0.90, n)
    multiple_lines    = RNG.binomial(1, 0.42, n) * phone_service
    internet_service  = RNG.choice([0, 1, 2], n, p=[0.22, 0.34, 0.44])
    online_security   = RNG.binomial(1, 0.38, n) * (internet_service > 0)
    online_backup     = RNG.binomial(1, 0.44, n) * (internet_service > 0)
    device_protection = RNG.binomial(1, 0.44, n) * (internet_service > 0)
    tech_support      = RNG.binomial(1, 0.33, n) * (internet_service > 0)
    streaming_tv      = RNG.binomial(1, 0.38, n) * (internet_service > 0)
    streaming_movies  = RNG.binomial(1, 0.39, n) * (internet_service > 0)

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
    support_calls = np.clip(
        RNG.poisson(lam=1.5 + (internet_service == 2) * 0.8, size=n), 0, 10
    )

    logit = (
        -3.5
        + 0.04  * (72 - tenure_months)
        + 0.015 * (monthly_charges - 60)
        - 1.20  * (contract == 1)
        - 2.10  * (contract == 2)
        + 0.25  * (internet_service == 2)
        - 0.40  * online_security
        - 0.35  * tech_support
        + 0.30  * support_calls
        + 0.18  * senior_citizen
        + 0.12  * (payment_method == 0)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, prob_churn, n)

    return pd.DataFrame({
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


if __name__ == "__main__":
    out = Path(__file__).parent / "churn_data.csv"
    df  = generate_dataset(5000)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows → {out}")
    print(f"Churn rate: {df['churn'].mean()*100:.1f}%")