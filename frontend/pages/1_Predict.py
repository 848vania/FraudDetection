import streamlit as st  

from frontend.utils import format_latency_ms, format_percent, post_prediction


st.set_page_config(
    page_title= "Predict",
    page_icon= "🔮",
    layout= "wide",
)

st.title("Fraud Risk Prediction")

st.write("Enter a transaction to score fraud risk.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_id = st.text_input("Transaction ID", "txn_demo_001")
        customer_id = st.text_input("Customer ID", "cust_demo_001")
        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value= 0.0,
            value= 249.99,
        )
        transaction_hour = st.slider("Transaction Hour", 0, 23, 23)
        merchant_category = st.selectbox(
            "Merchant Category",
            [
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
            ],
        )

    with col2:
        customer_tenure_days = st.number_input(
            "Customer Tenure Days",
            min_value= 0,
            value= 45,
        )
        num_transactions_24h = st.number_input(
            "Transactions in Last 24h",
            min_value= 0,
            value = 9,
        )
        num_failed_transactions_24h = st.number_input(
            "Failed Transactions in Last 24h",
            min_value= 0,
            value= 3,
        )
        avg_transaction_amount_30d = st.number_input(
            "Average Transaction Amount 30d",
            min_value= 0.0,
            value= 61.25,
        )
        previous_chargebacks = st.number_input(
            "Previous Chargebacks",
            min_value= 0,
            value=1,
        )

    with col3:
        is_foreign_transaction = st.checkbox(
            "Foreign Transaction",
            value= True,
        )
        device_type = st.selectbox(
            "Device Type",
            ["mobile", "desktop", "tablet", "pos"],
        )
        country = st.selectbox(
            "Country",
            ["US", "CA", "GB", "DE", "FR", "BR", "NG", "IN", "CN", "RU"],
        )
        account_age_days = st.number_input(
            "Account Age Days",
            min_value= 0,
            value=60,
        )
        risk_score_external = st.slider(
            "External Risk Score",
            0.0,
            1.0,
            0.71,
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        'transaction_id': transaction_id,
        'customer_id': customer_id,
        'transaction_amount': transaction_amount,
        'transaction_hour': transaction_hour,
        'merchant_category': merchant_category,
        'customer_tenure_days': customer_tenure_days,
        'num_transactions_24h': num_transactions_24h,
        'num_failed_transactions_24h': num_failed_transactions_24h,
        'avg_transaction_amount_30d': avg_transaction_amount_30d,
        'is_foreign_transaction': is_foreign_transaction,
        'device_type': device_type,
        'country': country,
        'previous_chargebacks': previous_chargebacks,
        'account_age_days': account_age_days,
        'risk_score_external': risk_score_external,
    }

    try:
        prediction = post_prediction(payload)

        st.subheader("Prediction Result")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Fraud Probability",
                format_percent(prediction['fraud_probability']),
            )

        with col2:
            st.metric("Risk Band", prediction['risk_band'])

        with col3:
            st.metric("Decision", prediction['decision'])

        with col4:
            st.metric(
                "Latency",
                format_latency_ms(prediction.get('latency_ms')),
            )

        st.write("Threshold used:", prediction.get('threshold_used'))
        st.write("Model:", prediction.get('model_name'))
        st.write("Version:", prediction.get('model_version'))
        st.write("Alias:", prediction.get('model_alias'))

        with st.expander("Raw prediction response"):
            st.json(prediction)

    except Exception as error:
        st.error(f"Prediction failed: {error}")