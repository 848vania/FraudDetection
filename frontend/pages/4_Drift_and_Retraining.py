import pandas as pd
import streamlit as st

from frontend.utils import (
    load_drift_summary,
    load_retraining_report,
    load_retraining_trigger_report,
)

st.set_page_config(
    page_title= 'Drift and Retraining',
    page_icon= "🔁",
    layout= 'wide',
)

st.title('Drift and Retraining')

drift_summary  = load_drift_summary()
trigger_report = load_retraining_trigger_report()
retraining_report = load_retraining_report()

if not drift_summary:
    st.warning("No drift summary found. Run 'python scripts/run_drift_check.py' first")
else:
    st.header('Drift Summary')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            'Overall Drift',
            'Yes' if drift_summary.get('overall_drift_detected') else 'No',
        )

    with col2:
        st.metric(
            'Drifted Features',
            drift_summary.get('num_drifted_features',0),
        )

    with col3:
        st.metric(
            'Recommendation',
            drift_summary.get('recommendation', '-'),
        )

    drifted_features = drift_summary.get('drifted_features', [])

    if drifted_features:
        st.subheader('Drifted Features')
        st.write(drifted_features)

    numeric_drift = drift_summary.get('numeric_drift', [])
    categorical_drift  = drift_summary.get('categorical_drift', [])

    if numeric_drift:
        st.subheader('Numeric Drift')
        st.dataframe(pd.DataFrame(numeric_drift), use_container_width= True)

    if categorical_drift:
        st.subheader('Categorical Drift')
        st.dataframe(pd.DataFrame(categorical_drift), use_container_width=True)

    with  st.expander('Raw drift summary'):
        st.json(drift_summary)

st.header('Retraining Trigger')

if not trigger_report:
    st.warning(
        "No retraining trigger report found. Run 'python scripts/trigger_retraining.py' first"
    )
else:
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            'Trigger Retraining',
            'Yes' if trigger_report.get('trigger_retraining') else 'No',
        )

    with col2:
        st.metric(
            'Recommended Action',
            trigger_report.get('recommended action', '-')
        )

    st.write('Reasons', trigger_report.get('reasons', []))

    with st.expander('Raw retraining trigger report'):
        st.json(trigger_report)

st.header('Retraining Pipeline')

if not retraining_report:
    st.info('No retraining report found yet')
else:
    st.metric(
        'Skipped',
        'Yes' if retraining_report.get('skipped') else 'No',
    )

    comparison = retraining_report.get('comparison')

    if comparison:
        st.subheader('Candidate vs Champion')
        st.json(comparison)

    registered_metadata = retraining_report.get('registered_metadata')

    if registered_metadata:
        st.subheader('Registered Candidate Metadata')
        st.json(registered_metadata)

    with st.expander('Raw retraining report'):
        st.json(retraining_report)