import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")

st.write("Upload a CSV file to detect fraudulent transactions.")

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

uploaded_file = st.file_uploader("Upload transaction file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "Class" in data.columns:
        X = data.drop("Class", axis=1)
    else:
        X = data

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    data["Prediction"] = predictions

    fraud_count = sum(predictions)

    st.success(f"Fraud Transactions Detected: {fraud_count}")
    st.dataframe(data.head())

else:
    st.info("Please upload a CSV file to begin.")
