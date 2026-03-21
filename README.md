# 🎯 Job Acceptance Prediction System

A Machine Learning project that predicts whether a candidate will accept or reject a job offer, built using real-world HR recruitment data.

---

## 📌 Project Overview

Recruitment teams face a major challenge — not every candidate who receives a job offer accepts it. This leads to wasted time, increased hiring costs, and longer recruitment cycles.

This project builds an end-to-end **Job Acceptance Prediction System** that:
- Predicts whether a candidate will **accept or reject** a job offer
- Identifies **key factors** influencing job acceptance decisions
- Provides **actionable insights** to improve recruitment strategies
- Delivers an **interactive dashboard** for HR managers

---

## 📂 Project Structure

```
Job-Acceptance-Prediction/
│
├── data/
│   ├── raw/                          # Original dataset
│   ├── cleaned/                      # Cleaned and processed data
│   └── featured/                     # Feature engineered data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Data cleaning and preprocessing
│   ├── 02_eda.ipynb                  # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb  # Feature engineering
│   └── 04_ml_modeling.ipynb          # Machine learning modeling
│
├── streamlit/
│   └── app.py                        # Interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Total Records:** 51,500 candidates
- **Features:** 26 columns covering academic, skills, experience, and job market data
- **Target:** `status` — Placed (1) or Not Placed (0)
- **Class Distribution:** 70% Not Placed, 30% Placed

### Feature Groups

| Group | Features |
|---|---|
| Academic | SSC %, HSC %, Degree % |
| Skills | Technical score, Aptitude score, Communication score, Skills match % |
| Experience | Years of experience, Internship experience, Relevant experience |
| Job Market | Company tier, Competition level, Job role match |
| Target | Status (Placed/Not Placed) |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting model |
| MySQL + SQLAlchemy | Data storage |
| Streamlit | Interactive dashboard |

---

## 🚀 Project Pipeline

### Step 1 — Data Understanding
- Loaded 51,500 candidate records
- Identified missing values across 6 columns
- Detected dirty data — case inconsistencies in gender, internship_experience
- Found trailing whitespace in company_tier
- Confirmed 70/30 class imbalance in target variable

### Step 2 — Data Cleaning and Preprocessing
- Fixed case inconsistencies using `str.strip().str.lower()`
- Filled numeric missing values with median
- Filled categorical missing values with mode
- Label encoded 7 binary columns
- Ordinal encoded 2 columns (company_tier, competition_level)
- One-hot encoded degree_specialization
- Applied StandardScaler on 14 numeric columns

### Step 3 — Exploratory Data Analysis
- Analyzed skills match percentage vs placement outcome
- Compared placement rates across company tiers
- Studied experience vs placement probability
- Examined competition level impact on acceptance
- Generated correlation heatmap for all features

### Step 4 — Feature Engineering
- Created `experience_category` (Fresher/Junior/Senior)
- Created `academic_performance_bands` from SSC, HSC, Degree average
- Created `skills_match_level` (Low/Average/Good)
- Created `interview_performance_category` from technical, aptitude, communication average
- Created `placement_probability` using weighted scoring formula

### Step 5 — Machine Learning Modeling
- Trained 3 classification models:
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Selected best model based on F1-Score

### Step 6 — SQL Storage
- Stored cleaned dataset in MySQL database
- Created structured tables with appropriate data types
- Enabled scalable querying using SQLAlchemy

### Step 7 — Streamlit Dashboard
- Built interactive KPI dashboard showing:
  - Total candidates
  - Placement rate
  - Average interview scores
  - Offer dropout rate
- Live prediction form for new candidates

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 78% | 0.76 | 0.74 | 0.75 | 0.82 |
| Random Forest | 85% | 0.84 | 0.83 | 0.83 | 0.91 |
| XGBoost | 87% | 0.86 | 0.85 | 0.85 | 0.93 |

**Best Model: XGBoost** with 87% accuracy and 0.93 ROC-AUC

---

## 🔑 Key Insights

- Candidates with **skills match > 75%** have 3x higher placement probability
- **Tier 1 companies** have significantly lower acceptance rates due to higher competition
- **Relevant experience** is the strongest predictor of job acceptance
- Candidates with **internship experience** are 40% more likely to be placed
- **High competition level** reduces placement probability by 35%

---

## ⚙️ How to Run

### 1. Clone the repository
```
git clone https://github.com/MIRUDHULA-DHANARAJ/Job-Acceptance-Prediction-System.git
cd Job-Acceptance-Prediction-System
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run notebooks in order
```
01_data_cleaning.ipynb
02_eda.ipynb
03_feature_engineering.ipynb
04_ml_modeling.ipynb
```

### 4. Launch Streamlit dashboard
```
streamlit run streamlit/app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
sqlalchemy
pymysql
streamlit
jupyter
```

---

## 🎯 Business Impact

| Before | After |
|---|---|
| 70% offer dropout rate | Predicted high-risk candidates early |
| No data-driven hiring | ML-powered acceptance prediction |
| Manual candidate evaluation | Automated scoring system |
| Reactive recruitment | Proactive placement strategy |

---

## 👩‍💻 Author

**Mirudhula Dhanaraj**
- GitHub: [@MIRUDHULA-DHANARAJ](https://github.com/MIRUDHULA-DHANARAJ)

---

## 📄 License

This project is for educational purposes as part of a Data Science capstone project.
