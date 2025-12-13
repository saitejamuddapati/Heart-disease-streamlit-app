# app.py
# Streamlit app for Heart Disease Predictor (RD INFRO TECHNOLOGY)
# This version uses friendly UI labels and maps them to the dataset's one-hot columns

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor (RD INFRO)", layout="centered")

# ---------------------- Wrapper & Helpers ----------------------
class ThresholdedModel:
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(X)
        else:
            scores = self.base_model.decision_function(X)
            from sklearn.preprocessing import MinMaxScaler
            probs = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()
            return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


def safe_load(path, name):
    if not os.path.exists(path):
        st.error(f"Required file missing: {path} ({name}). Place it in the app folder.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")
        st.stop()


# ---------------------- Friendly-to-short mappings ----------------------
CP_MAP = {
    "1 (Typical angina)": "cp_1",
    "2 (Atypical angina)": "cp_2",
    "3 (Non-anginal pain)": "cp_3",
    "4 (Asymptomatic)": "cp_1"
}
THAL_MAP = {
    "Normal": "thal_1",
    "Fixed defect": "thal_2",
    "Reversible defect": "thal_2"
}
SEX_MAP = {"Male": 1, "Female": 0}
EXANG_MAP = {"No": 0, "Yes": 1}

SCALER_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

MODEL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex_1", "cp_3", "exang_1", "thal_2"
]

# ---------------------- App UI ----------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("This app demonstrates the tuned logistic regression model.")

# ---------------------- Sidebar (MODIFIED) ----------------------
with st.sidebar:
    st.header("About This App")
    st.write(
        """
        This Heart Disease Risk Predictor is a machine learning–based application
        designed to estimate the likelihood of heart disease using common
        clinical parameters.

        It is intended for educational and awareness purposes, helping users
        understand potential cardiovascular risk based on data-driven analysis.
        """
    )

    st.markdown("---")

    st.header("How the App Works")
    st.markdown(
        """
        • You enter basic health and clinical details  
        • The app converts inputs into a machine-readable format  
        • Numeric values are scaled using the same preprocessing as training  
        • A tuned machine learning model evaluates the data  
        • You receive a probability score and risk classification  
        """
    )

# ---------------------- Load artifacts ----------------------
tuned = safe_load("best_model_tuned.joblib", "Final tuned model")
scaler = safe_load("scaler.joblib", "Scaler")
all_features = safe_load("feature_columns.joblib", "Saved feature list")

base_model = getattr(tuned, "base_model", tuned)
model_threshold = getattr(tuned, "threshold", 0.5)

# ---------------------- Inputs ----------------------
st.markdown("---")
st.header("Patient Input ")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 1, 120, 55)
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 50, 250, 140)
    chol = st.number_input("Serum cholesterol (mg/dL)", 50, 600, 240)
    thalach = st.number_input("Maximum heart rate achieved", 50, 250, 150)

with col2:
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.2, format="%.2f")
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest pain type",
        ["1 (Typical angina)", "2 (Atypical angina)", "3 (Non-anginal pain)", "4 (Asymptomatic)"]
    )
    exang = st.selectbox("Exercise-induced angina", ["No", "Yes"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

# ---------------------- Prediction ----------------------
st.markdown("---")

if st.button("Predict"):
    row = {f: 0.0 for f in all_features}

    row["age"] = age
    row["trestbps"] = trestbps
    row["chol"] = chol
    row["thalach"] = thalach
    row["oldpeak"] = oldpeak

    row["sex_1"] = 1.0 if sex == "Male" else 0.0
    row["exang_1"] = 1.0 if exang == "Yes" else 0.0

    cp_col = CP_MAP.get(cp)
    if cp_col in row:
        row[cp_col] = 1.0

    thal_col = THAL_MAP.get(thal)
    if thal_col in row:
        row[thal_col] = 1.0

    df_full = pd.DataFrame([row], columns=all_features)

    df_full[SCALER_COLS] = scaler.transform(df_full[SCALER_COLS])
    X_model = df_full[MODEL_FEATURES].values

    pred = tuned.predict(X_model)[0]
    prob = tuned.predict_proba(X_model)[0, 1]

    if prob >= 0.7:
        risk = "High"
    elif prob >= model_threshold:
        risk = "Medium"
    else:
        risk = "Low"

    st.subheader("Result")
    st.metric("Probability (positive class)", f"{prob:.3f}")
    st.write("**Prediction:**", "Positive — likely heart disease" if pred else "Negative — unlikely heart disease")
    st.write("**Risk level:**", risk)
    st.write("**Decision threshold used:**", model_threshold)

    if hasattr(base_model, "coef_"):
        coef_df = pd.DataFrame({
            "feature": MODEL_FEATURES,
            "coef": base_model.coef_.ravel()
        })
        st.subheader("Model coefficients")
        st.table(coef_df.set_index("feature"))

    out_df = df_full[MODEL_FEATURES].copy()
    out_df["probability"] = prob
    out_df["prediction"] = int(pred)

    st.download_button(
        "Download result (CSV)",
        out_df.to_csv(index=False),
        file_name="prediction_result.csv",
        mime="text/csv"
    )

    st.success("Prediction completed.")
