import streamlit as st
import pandas as pd
import joblib

model = joblib.load("valuation_under_over_xgboost.pkl")

MEDIAN_IMPORTER_FREQ = 50
MEDIAN_EXPORTER_FREQ = 50
MEDIAN_EXCHANGE_RATE = 0.5
MEDIAN_PREVIOUS_RISK = 0.5

SAMPLE_IMPORTER_FREQ = {
    "IMP001": 120,
    "IMP002": 80,
    "IMP003": 200
}

SAMPLE_EXPORTER_FREQ = {
    "EXP001": 150,
    "EXP002": 60,
    "EXP003": 180
}

def get_importer_freq(importer_id):
    return SAMPLE_IMPORTER_FREQ.get(importer_id, MEDIAN_IMPORTER_FREQ)

def get_exporter_freq(exporter_id):
    return SAMPLE_EXPORTER_FREQ.get(exporter_id, MEDIAN_EXPORTER_FREQ)

st.set_page_config(page_title="Valuation Risk Assessment", layout="centered")
st.markdown("<h2 style='text-align:center;'>Valuation Risk Assessment</h2>", unsafe_allow_html=True)

st.caption(
    "Note: Importer/Exporter historical frequency values shown here are illustrative placeholders for demo purposes."
)

with st.form("valuation_form"):
    importer_id = st.text_input("Importer ID")
    exporter_id = st.text_input("Exporter ID")
    declared_value = st.number_input("Declared Value", min_value=0.0)
    invoice_value = st.number_input("Invoice Value", min_value=0.0)
    assessed_value = st.number_input("Assessed Value", min_value=0.0)
    previous_risk = st.number_input(
        "Previous Overall Risk Score",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )
    submit = st.form_submit_button("Analyze")

if submit:
    importer_freq = get_importer_freq(importer_id)
    exporter_freq = get_exporter_freq(exporter_id)

    X = pd.DataFrame([{
        "invoice_value": invoice_value,
        "valuation_adjustment_amount": abs(assessed_value - declared_value),
        "declared_to_invoice_ratio": declared_value / (invoice_value + 1e-6),
        "declared_to_assessed_ratio": declared_value / (assessed_value + 1e-6),
        "assessed_value": assessed_value,
        "exchange_rate": MEDIAN_EXCHANGE_RATE,
        "declared_value": declared_value,
        "importer_freq": importer_freq,
        "declared_per_container": declared_value,
        "exporter_freq": exporter_freq,
        "risk_gap": previous_risk - MEDIAN_PREVIOUS_RISK,
        "previous_overall_risk_score": previous_risk
    }])

    prob = model.predict_proba(X)[0][1]

    st.markdown("---")
    if prob >= 0.6:
        st.markdown("<h3 style='text-align:center;'>Verdict: UNDER / OVER VALUED</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center;'>Verdict: NORMAL</h3>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='text-align:center;'>Risk Confidence: {prob*100:.1f}%</p>",
        unsafe_allow_html=True
    )
