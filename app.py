import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Valuation Risk Assessment", layout="centered")
st.markdown("<h2 style='text-align:center;'>Valuation Risk Assessment</h2>", unsafe_allow_html=True)

model = joblib.load("valuation_under_over_xgboost.pkl")

with st.form("valuation_form"):
    declared_value = st.number_input("Declared Value", min_value=0.0)
    invoice_value = st.number_input("Invoice Value", min_value=0.0)
    assessed_value = st.number_input("Assessed Value", min_value=0.0)
    previous_risk = st.number_input(
        "Previous Overall Risk Score",
        min_value=0.0,
        max_value=1.0,
        step=0.01
    )
    submit = st.form_submit_button("Analyze")

if submit:
    X = pd.DataFrame([{
        "invoice_value": invoice_value,
        "valuation_adjustment_amount": abs(assessed_value - declared_value),
        "declared_to_invoice_ratio": declared_value / (invoice_value + 1e-6),
        "declared_to_assessed_ratio": declared_value / (assessed_value + 1e-6),
        "assessed_value": assessed_value,
        "exchange_rate": 1.0,
        "declared_value": declared_value,
        "importer_freq": 1,
        "declared_per_container": declared_value,
        "exporter_freq": 1,
        "risk_gap": previous_risk - 1.0,
        "previous_overall_risk_score": previous_risk
    }])

    prediction = model.predict(X)[0]

    if prediction == 1:
        verdict = "UNDER / OVER VALUED"
    else:
        verdict = "NORMAL"

    st.markdown(f"### Verdict: {verdict}")
