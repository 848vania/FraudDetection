import pandas as pd 
import streamlit as st  

from frontend.utils import (
    format_latency_ms,
    format_percent,
    get_monitoring_summary,
)

st.set_page_config(
    page_title= 'Monitoring',
    page_icon="📈",
    layout='wide',
)

st.title('Prediction Monitoring')

try:
    summary = get_monitoring_summary()
except Exception as error:
    st.error(f'Failed to load monitoring summary: {error}')
    st.stop()

col1, col2, col3, col4, col5 = st.columns(5)

with col1: 
    st.metric('Total Predictions', summary['total_predictions'])

with col2:
    st.metric(
        'Avg latency',
        format_latency_ms(summary['average_latency_ms']),
    )

with col3:
    st.metric(
        'Review Rate',
        format_percent(summary['review_rate']),
    )

with col4:
    st.metric(
        'High Risk Rate',
        format_percent(summary['high_risk_rate']),
    )

with col5: 
    st.metric(
        'Avg Fraud Probability',
        format_percent(summary['average_fraud_probability']),
    )

st.header('Distributions')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Risk Band Distribution')
    st.bar_chart(pd.Series(summary['risk_band_distribution']))

with col2:
    st.subheader('Decision Distribution')
    st.bar_chart(pd.Series(summary['decision_distribution']))

col3, col4 = st.columns(2)

with col3:
    st.subheader('Model Version Distribution')
    st.bar_chart(pd.Series(summary['model_version_distribution']))

with col4:
    st.subheader('Source Distribution')
    st.bar_chart(pd.Series(summary['source_distribution']))

st.header('Prediction Volume By Day')

volume = summary.get('prediction_volume_by_day', {})
if volume:
    st.line_chart(pd.Series(volume))

st.header('Recent Predictions')

recent = summary.get('recent_predictions', [])

if recent:
    recent_df = pd.DataFrame(recent)

    visible_columns = [
        'created_at',
        'transaction_id',
        'fraud_probability',
        'risk_band',
        'decision',
        'model_version',
        'source',
        'latency_ms',
    ]

    visible_columns = [
        column for column in visible_columns
        if column in recent_df.columns
    ]

    st.dataframe(
        recent_df[visible_columns],
        use_container_width= True,
    )

    selected_index = st.number_input(
        'Inspect prediction index',
        min_value = 0,
        max_value= len(recent) -1,
        value = 0,
        step =1,
    )

    selected = recent[selected_index]

    with  st.expander('Selected prediction details', expanded=True):
        st.json(selected)