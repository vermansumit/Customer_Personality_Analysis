import streamlit as st
import joblib
import pandas as pd

st.title("Customer Response Predictor")
models = joblib.load("./models/random_forest_pipeline.joblib")

Income = st.number_input("Income", 0.0)
Age = st.number_input("Age", 18)
Recency = st.number_input("Recency (days since Last purchase)", 0)
TotalSpend = st.number_input("Total Spend", 0.0)
NumWebVisitsMonth = st.number_input("Num Web Visits/Month", 0)
Education = st.selectbox("Education", ["graduation", "phd", "master", "basic", "unknown"])
Marital_Status = st.selectbox("Marital Status", ["single", "married", "divorced", "unknown", "together", "widow"])

if st.button("Predict"):
    df = pd.DataFrame([{
        "Income": Income,
        "Age": Age,
        "Recency": Recency,
        "TotalSpend": TotalSpend,
        "NumWebVisitsMonth": NumWebVisitsMonth,
        "Education": Education,
        "Marital_Status": Marital_Status
    }])
    pred = models.predict(df)
    prob = float(models.predict_proba(df)[:,1])
    st.write(f"Predicted response: {pred} (prob {prob:.2f})")
    