import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Init SHAP JS
shap.initjs()

# Load model and data
model_bundle = joblib.load('./notebooks/models/random_forest.pkl')
rf_model = model_bundle['model']
X_sample = pd.read_csv('./notebooks/outputs/shap_input.csv')

# Recompute SHAP Explanation for class 1 (readmitted)
explainer = shap.Explainer(rf_model, X_sample)
shap_exp = explainer(X_sample)[..., 1]

# Title
st.set_page_config(layout="wide")
st.title("🩺 Explainable Health Optimizer Dashboard")
st.markdown("Visualizing diabetes readmission predictions using SHAP.")

# Sidebar: Patient selector
index = st.sidebar.slider("🔎 Select a patient index", 0, X_sample.shape[0] - 1, 0)

# =========================
# 🧬 Patient Data + Prediction
# =========================
st.subheader("🧬 Selected Patient Data")
st.dataframe(X_sample.iloc[[index]].T)

# Predict probability and label
proba = rf_model.predict_proba(X_sample.iloc[[index]])[0][1]
label = "🔴 At Risk of Readmission" if proba > 0.5 else "🟢 Not Likely to be Readmitted"
st.markdown(f"### 🎯 Predicted Probability: `{proba:.2f}`")
st.markdown(f"### 🧾 Predicted Label: **{label}**")

# =========================
# 🌍 Global Feature Importance
# =========================
st.subheader("🌍 Global SHAP Feature Importance (Top 15)")
shap.plots.bar(shap_exp, max_display=15, show=False)
st.pyplot(plt.gcf())

# =========================
# 📊 SHAP Summary Plot
# =========================
st.subheader("📊 SHAP Summary Plot")
shap.plots.beeswarm(shap_exp, max_display=15, show=False)
st.pyplot(plt.gcf())

# =========================
# 🔬 SHAP Force Plot
# =========================
st.subheader("🔬 Local SHAP Force Plot for Selected Patient (Static)")
plt.clf()
shap.force_plot(
    base_value=shap_exp.base_values[index],
    shap_values=shap_exp.values[index],
    features=X_sample.iloc[index],
    feature_names=X_sample.columns,
    matplotlib=True,
    show=False
)
st.pyplot(plt.gcf())
