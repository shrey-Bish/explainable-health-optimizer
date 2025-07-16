
# 🩺 Explainable Health Optimizer

A machine learning dashboard for predicting diabetes patient readmission and explaining predictions using SHAP (SHapley Additive exPlanations).

---

## 🚀 Project Overview

This project builds an interpretable ML pipeline that predicts whether a diabetic patient will be readmitted to a hospital within 30 days. It uses:

- A cleaned hospital records dataset
- A Random Forest classification model
- SHAP for global and local model explainability
- A Streamlit dashboard for interactive visualization

---

## 📁 Folder Structure

```

explainable-health-optimizer/
├── notebooks/
│   ├── 01\_data\_cleaning.ipynb
│   ├── 02\_model\_training.ipynb
│   ├── 03\_model\_explainability.ipynb
│   ├── models/
│   │   └── random\_forest.pkl
│   └── outputs/
│       ├── cleaned\_data.csv
│       ├── shap\_input.csv
│       └── shap\_values.npy
├── dashboard/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 📊 Features

- 🔍 View prediction probability & readmission risk for a patient
- 🧬 Visualize patient-specific SHAP force plots
- 🌍 Global feature importance (bar + beeswarm plots)
- 📄 Cleaned dataset and reproducible pipeline
- 🧪 ROC, confusion matrix, precision, accuracy (coming soon)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/explainable-health-optimizer.git
cd explainable-health-optimizer
````

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Run the Dashboard

```bash
streamlit run dashboard/app.py
```

* Use the sidebar to select patient index
* View local/global SHAP visualizations
* See prediction score and label

---

## 📚 Notebooks Included

| Notebook                        | Purpose                                          |
| ------------------------------- | ------------------------------------------------ |
| `01_data_cleaning.ipynb`        | Cleans raw hospital dataset and applies encoding |
| `02_model_training.ipynb`       | Trains and evaluates a Random Forest model       |
| `03_model_explainability.ipynb` | Generates SHAP values for model interpretation   |

---

## ✅ Requirements

* Python 3.9+
* `pandas`, `scikit-learn`, `shap`, `streamlit`, `matplotlib`, `joblib`

You can install everything via:

```bash
pip install -r requirements.txt
```

---

## 📌 To-Do / Future Improvements

* [x] Global & local explainability with SHAP
* [x] Prediction label + probability view
* [ ] Export to PDF per patient
* [ ] ROC AUC / confusion matrix panel
* [ ] File upload for new predictions
* [ ] Deploy to Streamlit Cloud

---

## ✍️ Author

**Shrey Bishnoi**
MS in Computer Science (Arizona State University)
[LinkedIn](https://www.linkedin.com/in/shrey-bishnoi/) | [GitHub](https://github.com/shrey-Bish)

---


### 📸 **Dashboard Visualizations**

Below are visual previews of the Explainable Health Optimizer in action:

---

#### 🏠 Main Dashboard View

<img src="assets/Dashboard.png" alt="Main Dashboard" width="800"/>

---

#### 🎯 Predicted Probability + Label

Displays predicted probability of readmission and the corresponding label.

<img src="assets/Probability_Predictor.png" alt="Prediction Section" width="700"/>

---

#### 🌍 Global SHAP Feature Importance

Highlights the most influential features across all predictions.

<img src="assets/Global_SHAP.png" alt="Global SHAP" width="700"/>

---

#### 📊 SHAP Summary Plot

Shows feature importance and distribution across the SHAP space.

<img src="assets/SHAP_Summary.png" alt="SHAP Summary" width="700"/>

---

#### 🔬 Local SHAP Force Plot

Explains why a specific patient was predicted as readmitted or not.

<img src="assets/Local_SHAP_Force_Plot.png" alt="Local SHAP Force Plot" width="800"/>

---


## 📄 License

MIT License – feel free to use, fork, and contribute.



