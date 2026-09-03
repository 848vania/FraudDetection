import streamlit as st 


st.set_page_config(
    page_title="Fraud Risk MLOps",
    page_icon="💳",
    layout= 'wide',
)

st.title("Fraud Risk MLOps System")

st.write(
    """
    End-to-End MLOps project for transaction fraud-risk prediction

    This demo includes model training, experiment tracking, threshold tuning,
    model registry, online inference, batch prediction, prediction logging,
    monitoring, drift detection, retraining triggers, and explainability.
    """
)

st.markdown(
    """
    ### Pages 

    **Predict**
    Submit a transaction and receive fraud probability, risk band, and decision.

    **Model Performance**
    Inspect evaluation metrics, threshold tuning, confusion matrix, and curves.

    **Monitoring**
    View live prediction logs, review rate, high-risk rate, latency, and model-version usage.

    **Drift and Retraining**
    Inspect data drift, prediction dirft, and retraining recommendations
    """
)