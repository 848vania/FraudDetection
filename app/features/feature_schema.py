TARGET_COLUMN = 'is_fraud'

ID_COLUMNS = [
    'transaction_id',
    'customer_id',
]

RAW_NUMERIC_COLUMNS = [
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

RAW_CATEGORICAL_COLUMNS =[
    'merchant_category',
    'device_type',
    'country',
    'is_foreign_transaction',
]

DERIVED_NUMERIC_COLUMNS = [
    'amount_to_customer_avg_ratio',
    'failed_transaction_rate_24h'
]

DERIVED_BINARY_COLUMNS = [
    'is_night_transaction',
    'high_velocity_flag',
    'new_acoount_flag',
    'has_previous_chargeback',
]

NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS + DERIVED_BINARY_COLUMNS

CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

REQUIRED_RAW_COLUMNS = (
    ID_COLUMNS
    + RAW_NUMERIC_COLUMNS
    + RAW_CATEGORICAL_COLUMNS
    + [TARGET_COLUMN]
)

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS

OUTPUT_COLUMNS = (
    ID_COLUMNS
    + FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

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