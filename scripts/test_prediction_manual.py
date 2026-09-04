import json

from app.models.predict import predict_transaction


def main():
    transaction = {
        'transaction_id': 'txn_manual_001',
        'customer_id': 'cust_manual_001',
        'transaction_amount': 249.99,
        'transaction_hour': 23,
        'merchant_category': 'electronics',
        'customer_tenure_days': 45,
        'num_transactions_24h': 9,
        'num_failed_transactions_24h': 3,
        'avg_transaction_amount_30d': 61.25,
        'is_foreign_transaction': True,
        'device_type': 'mobile',
        'country': 'US',
        'previous_chargebacks': 1,
        'account_age_days': 60,
        'risk_score_external': 0.71,
    }

    response = predict_transaction(transaction)

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()