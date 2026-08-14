"""Synthetic transaction data generation for the fraud-risk model.

Produces a labeled transaction dataset whose features and fraud label
follow the schema used across the rest of the project (see the online
prediction API's request schema). The fraud label is not assigned
independently of the features -- it is sampled from a logistic risk
score built out of the same signals a fraud model would rely on
(amount anomalies, velocity, account tenure, geography, merchant
category), so the resulting dataset carries a learnable signal rather
than pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "entertainment",
    "restaurant",
    "fashion",
    "utilities",
    "gambling",
    "crypto",
    "other",
]
MERCHANT_CATEGORY_WEIGHTS = [0.20, 0.12, 0.08, 0.10, 0.15, 0.12, 0.10, 0.05, 0.03, 0.05]
HIGH_RISK_CATEGORIES = {"gambling", "crypto", "electronics"}

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "BR", "NG", "IN", "CN", "RU"]
COUNTRY_WEIGHTS = [0.55, 0.08, 0.07, 0.06, 0.05, 0.05, 0.03, 0.05, 0.03, 0.03]

DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos"]
DEVICE_WEIGHTS = [0.50, 0.30, 0.10, 0.10]

NIGHT_HOURS = {0, 1, 2, 3, 4, 5}


@dataclass
class SyntheticDataConfig:
    """Parameters controlling synthetic transaction generation."""

    num_rows: int = 50_000
    fraud_rate: float = 0.03
    random_seed: int = 42
    output_path: str = "data/synthetic/transactions.csv"
    num_customers: int | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Binary-search an intercept so mean(sigmoid(logits + b)) == target_rate."""
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(logits + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _generate_customers(num_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    customer_tenure_days = rng.integers(1, 3651, size=num_customers)
    # The account is typically opened before the customer relationship is established
    # (e.g. sign-up precedes first purchase), so account age is tenure plus a gap.
    account_age_days = customer_tenure_days + rng.integers(0, 400, size=num_customers)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, num_customers + 1),
            "customer_tenure_days": customer_tenure_days,
            "account_age_days": account_age_days,
            "home_country": rng.choice(COUNTRIES, size=num_customers, p=COUNTRY_WEIGHTS),
            "baseline_amount": np.round(
                rng.lognormal(mean=3.8, sigma=0.9, size=num_customers), 2
            ),
            # Most customers have none; a small minority have a history of chargebacks.
            "previous_chargebacks": rng.poisson(lam=0.15, size=num_customers),
        }
    )


def generate_synthetic_transactions(config: SyntheticDataConfig) -> pd.DataFrame:
    """Generate a synthetic labeled transaction dataset as a DataFrame."""
    if not 0 < config.fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    rng = np.random.default_rng(config.random_seed)
    n = config.num_rows
    num_customers = config.num_customers or max(1, n // 5)

    customers = _generate_customers(num_customers, rng)
    # A small share of customers transact far more often than average (repeat shoppers).
    customer_weights = rng.pareto(a=2.5, size=num_customers) + 1.0
    customer_weights /= customer_weights.sum()
    txn_customer_idx = rng.choice(num_customers, size=n, p=customer_weights)
    txns = customers.iloc[txn_customer_idx].reset_index(drop=True)

    avg_transaction_amount_30d = np.round(
        np.clip(txns["baseline_amount"] * rng.normal(1.0, 0.15, size=n), 1.0, None), 2
    )

    # Most transactions are ordinary purchases scaled off the customer's own baseline;
    # a minority are amount outliers, which is one of the signals that drives risk.
    amount_multiplier = np.where(
        rng.random(n) < 0.06,
        rng.lognormal(mean=1.6, sigma=0.7, size=n),
        rng.lognormal(mean=0.0, sigma=0.35, size=n),
    )
    transaction_amount = np.round(
        np.clip(avg_transaction_amount_30d * amount_multiplier, 1.0, None), 2
    )

    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n, p=MERCHANT_CATEGORY_WEIGHTS)
    device_type = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_WEIGHTS)

    num_transactions_24h = rng.poisson(lam=1.5, size=n)
    num_transactions_24h = np.clip(num_transactions_24h, 0, None)
    # Failed attempts are rare in general but spike for card-testing style fraud.
    num_failed_transactions_24h = np.clip(rng.poisson(lam=0.25, size=n), 0, None)

    transaction_hour = rng.integers(0, 24, size=n)

    transaction_country = np.where(
        rng.random(n) < 0.92,
        txns["home_country"].to_numpy(),
        rng.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS),
    )
    is_foreign_transaction = transaction_country != txns["home_country"].to_numpy()

    # --- Risk score driving the fraud label -------------------------------
    amount_ratio = transaction_amount / (avg_transaction_amount_30d + 1.0)
    log_amount_ratio = np.log(amount_ratio)
    tenure_days = txns["customer_tenure_days"].to_numpy()
    account_age_days = txns["account_age_days"].to_numpy()
    previous_chargebacks = txns["previous_chargebacks"].to_numpy()
    is_high_risk_category = np.isin(merchant_category, list(HIGH_RISK_CATEGORIES))
    is_night_hour = np.isin(transaction_hour, list(NIGHT_HOURS))

    logits = (
        1.6 * log_amount_ratio
        + 0.35 * num_transactions_24h
        + 0.9 * num_failed_transactions_24h
        - 0.004 * tenure_days
        - 0.002 * account_age_days
        + 0.5 * previous_chargebacks
        + 1.1 * is_foreign_transaction
        + 0.9 * is_high_risk_category
        + 0.4 * is_night_hour
        + rng.normal(0.0, 0.6, size=n)  # unexplained noise
    )
    intercept = _calibrate_intercept(logits, config.fraud_rate)
    fraud_probability = _sigmoid(logits + intercept)
    is_fraud = (rng.random(n) < fraud_probability).astype(int)

    # A third-party risk score: correlated with the true risk signal but noisier
    # and on its own 0-1 scale, as a vendor-supplied feature would be.
    risk_score_external = np.clip(
        _sigmoid(0.6 * logits + rng.normal(0.0, 1.2, size=n)), 0.0, 1.0
    ).round(4)

    data = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n + 1),
            "customer_id": txns["customer_id"].to_numpy(),
            "transaction_amount": transaction_amount,
            "transaction_hour": transaction_hour,
            "merchant_category": merchant_category,
            "customer_tenure_days": tenure_days,
            "num_transactions_24h": num_transactions_24h,
            "num_failed_transactions_24h": num_failed_transactions_24h,
            "avg_transaction_amount_30d": avg_transaction_amount_30d,
            "is_foreign_transaction": is_foreign_transaction,
            "device_type": device_type,
            "country": transaction_country,
            "previous_chargebacks": previous_chargebacks,
            "account_age_days": account_age_days,
            "risk_score_external": risk_score_external,
            "is_fraud": is_fraud,
        }
    )
    return data


def save_synthetic_transactions(config: SyntheticDataConfig) -> Path:
    """Generate synthetic transactions and write them to config.output_path as CSV."""
    df = generate_synthetic_transactions(config)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
