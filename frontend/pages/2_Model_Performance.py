import streamlit as st

from frontend.utils import (
    format_cost,
    format_percent,
    load_evaluation_metrics,
    load_threshold_report,
)


st.set_page_config(
    page_title= 'Model Performance',
    page_icon="📊",
    layout= 'wide',
)

st.title('Model Performance')

metrics = load_evaluation_metrics()
threshold_report = load_threshold_report()

if not metrics:
    st.warning(
        "No evaluation metrics found. Run 'python scripts/evaluate_model.py' first"
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0):.3f}")

with col2:
    st.metric('Test PR-AUC', f"{metrics.get('test_pr_auc', 0):.3f}")

with col3:
    st.metric('Precision', format_percent(metrics.get('precision')))

with col4:
    st.metric('Recall', format_percent(metrics.get('recall')))


col5, col6, col7, col8  = st.columns(4)

with col5:
    st.metric('F1', f"{metrics.get('f1',0):.3f}")

with col6:
    st.metric('Review Rate', format_percent(metrics.get('review_rate')))

with col7:
    st.metric('Expected Cost', format_cost(metrics.get('expected_cost')))

with col8:
    st.metric('Threshold', f"{metrics.get('selected_threshold',0):.2f}")

st.header('Confusion Matrix and Curves')

col1, col2, col4 = st.columns(3)

with col1:
    path = 'data/results/confusion_matrix.png'
    st.image(path, caption='Confusion Matrix')

with col2:
    path = 'data/results/precision_recall_curve.png'
    st.image(path, caption='Precision-Recall Curve')

with col3:
    path = 'data/results/roc_curve.png'
    st.image(path, caption='ROC Curve')

st.header('Threshold Tuning')

if threshold_report:
    best  = threshold_report.get('best_threshold_metrics', {})

    st.write('Objective:', threshold_report.get('objective'))
    st.write('Best threshold:', threshold_report.get('best_threshold'))

    st.dataframe(
        {
            'metric': list(best.keys()),
            'value': list(best.values()),
        },
        use_container_width= True
    )

with st.expander('Raw evaluation metrics'):
    st.json(metrics)

with st.expander('Raw Threshold  report'):
    st.json(threshold_report)