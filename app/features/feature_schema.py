TARGET_COLUMN = 'is_fraud'

ID_COLUMNS = [
    'transaction_id',
    'customer_id',
]

NUMERIC_COLUMNS = [
    'transaction_amount',
    'transaction_hour',
    'customer_tenure_days',
    'num_transactions_24h',
    'num_failed_transactions_24h',
    'avg_transaction_amount_30d',
    'previous_chargebacks',
    'account_age_days',
    'risk_score_external',
]

CATEGORICAL_COLUMNS =[
    'merchant_category',
    'device_type',
    'country',
    'is_foreign_transaction',
]

REQUIRED_COLUMNS = (
    ID_COLUMNS
    + NUMERIC_COLUMNS
    + CATEGORICAL_COLUMNS
    + [TARGET_COLUMN]
)

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

ALLOWED_VALUES = {
    'merchant_category': [
        'grocery',
        'electronics',
        'travel',
        'entertainment',
        'restaurant',
        'fashion',
        'utilities',
        'gambling',
        'crypto',
        'other',
    ],
    'device_type': [
        'mobile',
        'desktop',
        'tablet',
        'pos',
    ],
    'country': [
        'US',
        'CA',
        'GB',
        'FR',
        'DE',
        'BR',
        'NG',
        'IN',
        'CN',
        'RU',
        'OTHER',
    ],
    'is_foreign_transaction': [
        True,
        False,
        0,
        1,
    ],
}

NUMERIC_RANGES = {
    'transaction_amount': {
        'min': 0.0,
        'max': 100000.0,
    },
    'transaction_hour': {
        'min': 0,
        'max': 23,
    },
    'customer_tenure_days': {
        'min': 0,
        'max': 36500
    },
    'num_transactions_24h': {
        'min': 0,
        'max': 1000,
    },
    'num_failed_transactions_24h': {
        'min': 0,
        'max': 1000,
    },
    'avg_transaction_amount_30d': {
        'min': 0.0,
        'max': 100000.0,
    },
    'previous_chargebacks': {
        'min': 0,
        'max': 100,
    },
    'account_age_days': {
        'min': 0,
        'max': 36500,
    },
    'risk_score_external': {
        'min': 0.0,
        'max': 1.0,
    },
}

TARGET_ALLOWED_VALUES = [0, 1]