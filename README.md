# 🏆 Engineering Job Placement Predictive Super App

**Capstone Project | End-to-End Machine Learning Engineering Pipeline**

A full-stack, production-aware ML system that ingests raw industrial candidate data, tracks experiments via MLflow, and serves real-time placement predictions through a premium glassmorphic Streamlit dashboard.

🔗 **Live Demo:** [Insert your Streamlit Share link here]

---

## 📌 Problem Statement

Placement teams and engineering institutions face real challenges in predicting student outcomes:

- No systematic way to identify which candidates are likely to accept a job offer
- Manual shortlisting is slow, biased, and inconsistent across branches
- Imbalanced placement datasets lead to misleading accuracy metrics
- No experiment tracking — every model retrain is a black box
- Scaling transformations applied before splitting cause data leakage and overfit models

This project solves all of it — systematically.

---

## 🎯 Domain

**Talent Analytics & Placement Intelligence System**

---

## 🖥️ App Overview

| Page | Description |
|---|---|
| 🏠 Home | Project summary, pipeline overview, quick-start guide |
| 🔮 Predict | Real-time single-candidate placement prediction with probability score |
| 📊 Dashboard | Feature importance, model comparison chart, ROC-AUC curve |
| 🧪 MLflow Tracker | Experiment leaderboard — all runs, metrics, and parameters |

---

## 📁 Repository Structure

```
Job-Acceptance-Prediction-System/
│
├── models/
│   ├── scaler.pkl               # Trained StandardScaler (fit on train set only)
│   └── feature_columns.pkl      # Serialized feature mask — the data contract
│
├── src/
│   ├── data_cleaning.py         # Deduplication, null handling, structural sanitation
│   ├── feature_engineering.py   # Stratified splits, encoding, leakage guardrails
│   └── train.py                 # Model training + MLflow experiment logging
│
├── app_ui.py                    # Glassmorphic Streamlit prediction dashboard
├── requirements.txt             # Version-pinned dependencies
└── README.md                    # You are here
```

---

## 🗄️ ML Pipeline Design

### Data Flow

```
Raw CSV
  │
  ▼
data_cleaning.py       ← Remove duplicates, fix nulls, drop anomalies
  │
  ▼
feature_engineering.py ← Stratified Train/Val/Test split → encode → scale (train only)
  │
  ├── scaler.pkl                 (saved)
  └── feature_columns.pkl        (saved)
  │
  ▼
train.py               ← Train Logistic Regression → Random Forest → XGBoost (champion)
  │
  └── MLflow (SQLite backend)    (all runs logged)
  │
  ▼
app_ui.py              ← Load artifacts → accept UI input → predict → display
```

### Models Trained & Tracked

| Model | Logged Metrics |
|---|---|
| Logistic Regression | Accuracy, F1, ROC-AUC, Precision, Recall |
| Random Forest | Accuracy, F1, ROC-AUC, Precision, Recall |
| ✅ XGBoost (Champion) | Accuracy, F1, ROC-AUC, Precision, Recall |

---

## 🛡️ Production Engineering Guardrails

These are fully defensible design decisions for technical interviews:

**Strict Data Leakage Prevention**
The dataset is partitioned into Train / Validation / Test subsets *before* any transformation. The `StandardScaler` is fit exclusively on the training partition. Validation and Test sets are only transformed — never used to fit anything. This guarantees honest generalization metrics.

**Schema Contract at Inference**
The `feature_columns.pkl` artifact is the data contract between training and serving. At prediction time, the UI input is force-aligned against this saved schema. Missing or unexpected categorical values are auto-filled with zeros. The app will never crash due to a column mismatch between training and production.

**Imbalance-Aware Evaluation**
Model selection uses F1-Score and ROC-AUC — not raw accuracy. Raw accuracy on imbalanced placement data can be misleadingly high by predicting the majority class. F1 and AUC give a true picture of classifier quality across both classes.

**Stateful Cache Optimization**
`scaler.pkl` and `feature_columns.pkl` are loaded once at startup using `@st.cache_resource`. Streamlit reruns the script on every user interaction — without caching, this would trigger repeated disk reads. The cache ensures artifacts are loaded into memory exactly once per session.

**Single Source of Truth for Metrics**
All experiment results are logged to a SQLite-backed MLflow store (`mlflow.db`). No metrics live in notebooks, print statements, or local variables. Every run is reproducible and comparable in the MLflow UI.

---

## 🐛 Real Bugs Fixed During Development

This project went through real debugging. Here's what broke and how it was fixed:

**Bug #1 — Data Leakage Through the Scaler**
*Symptom:* Validation accuracy was suspiciously high — model appeared to generalise perfectly.
*Root Cause:* `StandardScaler` was fit on the entire dataset before splitting. The scaler had already "seen" the validation and test distributions during training.
*Fix:* Moved `scaler.fit()` to run only on `X_train`. Validation and test sets use `scaler.transform()` only. Metrics dropped to honest values.

**Bug #2 — App Crash on Unseen Categorical Values**
*Symptom:* `KeyError` in production when the UI submitted a category not present in training data.
*Root Cause:* One-hot encoding during inference produced different columns than during training. The model received a misaligned input vector.
*Fix:* Saved `feature_columns.pkl` after training. At inference, the UI input DataFrame is reindexed against this saved column list — missing columns are filled with 0, extra columns are dropped.

**Bug #3 — MLflow Logging Conflicting Across Runs**
*Symptom:* Metrics from one experiment appeared under a different run in the MLflow UI.
*Root Cause:* `mlflow.start_run()` was not being explicitly closed. Nested calls shared the same active run context.
*Fix:* Wrapped every training block in a `with mlflow.start_run():` context manager. Each run is now isolated and closed automatically.

**Bug #4 — Streamlit Caching Stale Artifacts After Retrain**
*Symptom:* Predictions in the UI did not change after retraining and saving a new `scaler.pkl`.
*Root Cause:* `@st.cache_resource` held the old artifact in memory. Streamlit served the cached version even after the file on disk changed.
*Fix:* Added a manual `st.cache_resource.clear()` call in the dev workflow after every retrain. In production, a clean app restart clears the cache.

---

## ⚙️ Features

### 🔮 Real-Time Prediction
- Single candidate input via sidebar form
- Prediction output: **Placed** or **Not Placed**
- Probability confidence score displayed with a progress bar
- Input auto-aligned to training schema via `feature_columns.pkl`

### 📊 Model Analytics Dashboard
- Feature importance bar chart (XGBoost)
- ROC-AUC curve with AUC score
- Model comparison leaderboard (all 3 models side by side)
- Confusion matrix heatmap

### 🧪 Experiment Tracking (MLflow)
- All runs logged to `mlflow.db` (SQLite backend)
- Parameters: `n_estimators`, `max_depth`, `learning_rate`, etc.
- Metrics: Accuracy, F1, Precision, Recall, ROC-AUC
- Launch the UI locally with one command (see below)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| ML Models | XGBoost, Scikit-Learn |
| Experiment Tracking | MLflow + SQLite |
| Data Processing | Pandas, NumPy |
| Serialization | Joblib |
| Charts | Plotly Express |
| UI Styling | Custom CSS (Glassmorphic dark theme) |
| Artifact Caching | `@st.cache_resource` |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Job-Acceptance-Prediction-System.git
cd Job-Acceptance-Prediction-System
```

### 2. Set Up the Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Data Processing

```bash
# Clean raw data — remove duplicates, fix nulls, resolve structural issues
python src/data_cleaning.py

# Stratified split → encode → scale → export scaler.pkl + feature_columns.pkl
python src/feature_engineering.py
```

### 4. Train Models & Log Experiments

```bash
# Train Logistic Regression → Random Forest → XGBoost
python src/train.py

# Launch the MLflow experiment tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://localhost:5000
```

### 5. Launch the Prediction Dashboard

```bash
streamlit run app_ui.py
# Open: http://localhost:8501
```

---

## 💡 Key Engineering Decisions

**Why stratified splits instead of random splits?**
The target class (Placed / Not Placed) is imbalanced. A random split risks putting almost all positive examples in one partition. Stratified splitting preserves the class ratio across Train, Validation, and Test — ensuring each partition reflects the real distribution.

**Why save `feature_columns.pkl` separately from the model?**
The trained model only knows about the column indices it saw during training. Saving the column list as a separate artifact creates an explicit, inspectable data contract between the training environment and the serving environment. Any mismatch is caught and corrected at inference time — not at crash time.

**Why XGBoost as the champion over Random Forest?**
XGBoost natively handles class imbalance via the `scale_pos_weight` parameter, supports early stopping to prevent overfit, and logged the highest ROC-AUC across all experiment runs in MLflow. The comparison is documented — not assumed.

**Why MLflow with SQLite instead of the default file store?**
The default MLflow file store writes metrics as flat files. SQLite provides a queryable backend — useful for comparing runs programmatically and for future integration with a hosted tracking server without changing the logging code.

---

## 📊 Dataset Overview

| Feature | Type | Description |
|---|---|---|
| CGPA | Numeric | Academic performance score |
| Internship | Categorical | Completed internship (Yes/No) |
| Projects | Numeric | Number of projects completed |
| Workshops | Numeric | Workshops / certifications attended |
| AptitudeScore | Numeric | Aptitude test percentile |
| SoftSkillRating | Numeric | Communication & soft skills rating |
| ExtraCurricular | Categorical | Participated in activities (Yes/No) |
| PlacementTraining | Categorical | Attended placement prep (Yes/No) |
| SSC_Marks | Numeric | Class 10 marks |
| HSC_Marks | Numeric | Class 12 marks |
| **PlacementStatus** | **Target** | **Placed / Not Placed** |

---

## 📋 Project Deliverables

| Deliverable | Status |
|---|---|
| Raw data cleaning script | ✅ |
| Leakage-safe feature engineering | ✅ |
| Multi-model training with MLflow logging | ✅ |
| Serialized scaler + feature schema artifacts | ✅ |
| Real-time Streamlit prediction UI | ✅ |
| Glassmorphic dark theme dashboard | ✅ |
| Experiment leaderboard (MLflow UI) | ✅ |
| Documentation (this README) | ✅ |

---

## 👨‍💻 Author

**[Your Name]** — [Your Role / Program]

- 🎓 [Your program / institution]
- 💼 [Your background or experience]
- 📧 [your.email@example.com]
- 🔗 [GitHub Profile](https://github.com/YOUR_USERNAME)
- 📍 [Your city] | Open to [opportunities]

---

## 🙏 Acknowledgement

[Your mentor / guide / institution shoutout here.]

---

## 🏷️ Technical Tags

`Python` `XGBoost` `Scikit-Learn` `MLflow` `Streamlit` `Pandas` `NumPy` `Plotly` `Joblib` `SQLite` `Machine Learning` `Classification` `Placement Prediction` `Feature Engineering` `Experiment Tracking` `Data Leakage Prevention` `End-to-End ML Pipeline`

---

## 📄 License

This project is open source and available under the MIT License.
