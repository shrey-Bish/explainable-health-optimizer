import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

# Feature name mapping
feature_name_mapping = {
    "age": "Age",
    "admission_type_id": "Admission Type",
    "discharge_disposition_id": "Discharge Disposition",
    "admission_source_id": "Admission Source",
    "time_in_hospital": "Days in Hospital",
    "num_lab_procedures": "Lab Procedures",
    "num_procedures": "Procedures Performed",
    "num_medications": "Medications Given",
    "number_outpatient": "Outpatient Visits",
    "number_emergency": "Emergency Visits",
    "number_inpatient": "Inpatient Visits",
    "number_diagnoses": "Number of Diagnoses",

    # Race
    "race_Asian": "Race: Asian",
    "race_Caucasian": "Race: Caucasian",
    "race_Hispanic": "Race: Hispanic",
    "race_Other": "Race: Other",

    # Gender
    "gender_Male": "Gender: Male",
    "gender_Unknown/Invalid": "Gender: Unknown",

    # Medications - Metformin and others
    "metformin_No": "Metformin: Not Used",
    "metformin_Steady": "Metformin: Steady Dose",
    "metformin_Up": "Metformin: Increased Dose",

    "repaglinide_No": "Repaglinide: Not Used",
    "repaglinide_Steady": "Repaglinide: Steady Dose",
    "repaglinide_Up": "Repaglinide: Increased Dose",

    "nateglinide_No": "Nateglinide: Not Used",
    "nateglinide_Steady": "Nateglinide: Steady Dose",
    "nateglinide_Up": "Nateglinide: Increased Dose",

    "chlorpropamide_No": "Chlorpropamide: Not Used",
    "chlorpropamide_Steady": "Chlorpropamide: Steady Dose",
    "chlorpropamide_Up": "Chlorpropamide: Increased Dose",

    "glimepiride_No": "Glimepiride: Not Used",
    "glimepiride_Steady": "Glimepiride: Steady Dose",
    "glimepiride_Up": "Glimepiride: Increased Dose",

    "acetohexamide_Steady": "Acetohexamide: Steady Dose",

    "glipizide_No": "Glipizide: Not Used",
    "glipizide_Steady": "Glipizide: Steady Dose",
    "glipizide_Up": "Glipizide: Increased Dose",

    "glyburide_No": "Glyburide: Not Used",
    "glyburide_Steady": "Glyburide: Steady Dose",
    "glyburide_Up": "Glyburide: Increased Dose",

    "tolbutamide_Steady": "Tolbutamide: Steady Dose",

    "pioglitazone_No": "Pioglitazone: Not Used",
    "pioglitazone_Steady": "Pioglitazone: Steady Dose",
    "pioglitazone_Up": "Pioglitazone: Increased Dose",

    "rosiglitazone_No": "Rosiglitazone: Not Used",
    "rosiglitazone_Steady": "Rosiglitazone: Steady Dose",
    "rosiglitazone_Up": "Rosiglitazone: Increased Dose",

    "acarbose_No": "Acarbose: Not Used",
    "acarbose_Steady": "Acarbose: Steady Dose",
    "acarbose_Up": "Acarbose: Increased Dose",

    "miglitol_No": "Miglitol: Not Used",
    "miglitol_Steady": "Miglitol: Steady Dose",
    "miglitol_Up": "Miglitol: Increased Dose",

    "troglitazone_Steady": "Troglitazone: Steady Dose",

    "tolazamide_Steady": "Tolazamide: Steady Dose",
    "tolazamide_Up": "Tolazamide: Increased Dose",

    "insulin_No": "Insulin: Not Used",
    "insulin_Steady": "Insulin: Steady Dose",
    "insulin_Up": "Insulin: Increased Dose",

    "glyburide-metformin_No": "Glyburide-Metformin: Not Used",
    "glyburide-metformin_Steady": "Glyburide-Metformin: Steady Dose",
    "glyburide-metformin_Up": "Glyburide-Metformin: Increased Dose",

    "glipizide-metformin_Steady": "Glipizide-Metformin: Steady Dose",
    "glimepiride-pioglitazone_Steady": "Glimepiride-Pioglitazone: Steady Dose",
    "metformin-rosiglitazone_Steady": "Metformin-Rosiglitazone: Steady Dose",
    "metformin-pioglitazone_Steady": "Metformin-Pioglitazone: Steady Dose",

    "change_No": "Change in Medications: No",
    "diabetesMed_Yes": "On Diabetes Medication"
}


def map_feature_names(cols):
    return [feature_name_mapping.get(col, col) for col in cols]

# Setup page
st.set_page_config(layout="wide", page_title="Explainable Health Optimizer")
st.title("🩺 Explainable Health Optimizer Dashboard")
st.markdown("Visualizing diabetes readmission predictions using **SHAP** & **LightGBM**.")

# Set up base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'notebooks', 'models', 'lightgbm.pkl')
INPUT_PATH = os.path.join(BASE_DIR, 'notebooks', 'assets', 'shap_input.csv')

# Load model and data
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle['model']
X_sample = pd.read_csv(INPUT_PATH)

# SHAP calculation
explainer = shap.Explainer(model, X_sample)
shap_values_full = explainer(X_sample)

# Binary / Multiclass
shap_values_class1 = shap_values_full[..., 1] if len(shap_values_full.shape) == 3 else shap_values_full

# Sidebar: Patient selector
st.sidebar.header("🧬 Patient Selection")
index = st.sidebar.slider("Select a patient index", 0, X_sample.shape[0] - 1, 0)

# =========================
# Patient Info + Prediction
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("👤 Selected Patient Data")
    patient_df = X_sample.iloc[[index]].T
    patient_df.index = map_feature_names(patient_df.index)
    st.dataframe(patient_df)

with col2:
    proba = model.predict_proba(X_sample.iloc[[index]])[0][1]
    label = "🔴 At Risk of Readmission" if proba > 0.5 else "🟢 Not Likely to be Readmitted"

st.markdown(f"### 🎯 Predicted Probability: `{proba:.2f}`")

if proba > 0.5:
    st.error(f"🧾 Predicted Label: 🔴 At Risk of Readmission")
else:
    st.success(f"🧾 Predicted Label: 🟢 Not Likely to be Readmitted")

    st.subheader("🎚️ Model Confidence")
    st.progress(proba)

# =========================
# Global SHAP Importance
# =========================
with st.expander("🌍 Global Feature Importance (Top 15)", expanded=True):
    plt.clf()
    shap.plots.bar(shap_values_class1, max_display=15, show=False)
    st.pyplot(plt.gcf())

with st.expander("📊 SHAP Summary Beeswarm Plot", expanded=False):
    plt.clf()
    shap.plots.beeswarm(shap_values_class1, max_display=15, show=False)
    st.pyplot(plt.gcf())

# =========================
# Local Explanation: Force + Waterfall
# =========================
with st.expander("🔬 Local SHAP Force Plot"):
    plt.clf()
    shap.force_plot(
        base_value=shap_values_class1.base_values[index],
        shap_values=shap_values_class1.values[index],
        features=X_sample.iloc[index],
        feature_names=X_sample.columns,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())

with st.expander("🌊 SHAP Waterfall Plot"):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values_class1[index], max_display=15, show=False)
    st.pyplot(fig)

# =========================
# Top Feature Contributions
# =========================
with st.expander("📌 Top 5 Contributing Features", expanded=True):
    shap_values_patient = shap_values_class1[index].values
    feature_contributions = pd.DataFrame({
        "Feature": map_feature_names(X_sample.columns),
        "SHAP Value": shap_values_patient,
        "Feature Value": X_sample.iloc[index].values
    })
    top_features = feature_contributions.reindex(
        feature_contributions["SHAP Value"].abs().sort_values(ascending=False).index
    ).head(5)
    st.dataframe(top_features.set_index("Feature").style.format({
        "SHAP Value": "{:.4f}",
        "Feature Value": "{:.2f}"
    }))

# =========================
# SHAP Dependence Plot
# =========================
with st.expander("📈 SHAP Dependence Plot"):
    col_display_names = map_feature_names(X_sample.columns)
    selected_display_name = st.selectbox("🔬 Select feature", col_display_names)

    # Map display name back to original column
    inverse_mapping = {v: k for k, v in feature_name_mapping.items()}
    selected_feature = inverse_mapping.get(selected_display_name, selected_display_name)

    if selected_feature:
        plt.clf()
        shap.dependence_plot(selected_feature, shap_values_class1.values, X_sample, show=False)
        st.pyplot(plt.gcf())

# =========================
# Patient Report Download
# =========================
with st.expander("📥 Downloadable Patient Report"):
    report_dict = {
        "Patient Index": index,
        "Predicted Probability": proba,
        "Predicted Label": label,
    }
    top_k = 5
    top_indices = np.argsort(np.abs(shap_values_class1.values[index]))[::-1][:top_k]
    for i, idx in enumerate(top_indices):
        feature = X_sample.columns[idx]
        feature_name = feature_name_mapping.get(feature, feature)
        value = X_sample.iloc[index, idx]
        shap_val = shap_values_class1.values[index, idx]
        report_dict[f"Feature {i+1}"] = feature_name
        report_dict[f"Value {i+1}"] = value
        report_dict[f"SHAP {i+1}"] = shap_val

    report_df = pd.DataFrame([report_dict])
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Report as CSV", data=csv,
                       file_name=f"patient_{index}_report.csv", mime='text/csv')
