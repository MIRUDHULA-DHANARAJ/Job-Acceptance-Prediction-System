import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page config
st.set_page_config(
    page_title="Job Acceptance Prediction",
    page_icon="🎯",
    layout="wide"
)

# Load data and model
df = pd.read_csv("data/cleaned/cleaned_unscaled.csv")
model = joblib.load("models/xgboost_model.pkl")

# Title
st.title("🎯 Job Acceptance Prediction System")
st.markdown("HR Analytics Dashboard")
## Run It First

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Candidates", len(df))

with col2:
    placement_rate = round(df['status'].mean() * 100, 1)
    st.metric("Placement Rate", f"{placement_rate}%")

with col3:
    avg_score = round(df['technical_score'].mean(), 1)
    st.metric("Avg Technical Score", avg_score)

with col4:
    dropout_rate = round((1 - df['status'].mean()) * 100, 1)
    st.metric("Dropout Rate", f"{dropout_rate}%")

st.subheader("📊 Placement Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df['status'].value_counts().plot.pie(
        labels=['Not Placed', 'Placed'],
        autopct='%1.1f%%',
        ax=ax
    )
    ax.set_title('Placed vs Not Placed')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='company_tier', y='status', ax=ax)
    ax.set_title('Company Tier vs Placement Rate')
    ax.set_xlabel('Company Tier')
    ax.set_ylabel('Placement Rate')
    st.pyplot(fig)

st.subheader("🎯 Live Job Acceptance Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 21, 35, 25)
    technical_score = st.slider("Technical Score", 40, 100, 70)
    aptitude_score = st.slider("Aptitude Score", 40, 100, 70)
    communication_score = st.slider("Communication Score", 40, 100, 70)
    skills_match = st.slider("Skills Match %", 40, 100, 70)

with col2:
    ssc = st.slider("SSC %", 50, 95, 70)
    hsc = st.slider("HSC %", 50, 95, 70)
    degree = st.slider("Degree %", 55, 95, 70)
    experience = st.selectbox("Years of Experience", [0,1,2,3,4,5])
    certifications = st.selectbox("Certifications Count", [0,1,2,3,4,5])

with col3:
    internship = st.selectbox("Internship Experience", [0,1])
    relevant_exp = st.selectbox("Relevant Experience", [0,1])
    company_tier = st.selectbox("Company Tier", [1,2,3])
    competition = st.selectbox("Competition Level", [0,1,2])
    job_role_match = st.selectbox("Job Role Match", [0,1])

if st.button("🎯 Predict Job Acceptance"):
    
    # Create input array
    input_data = pd.DataFrame({
        'age_years': [age],
        'ssc_percentage': [ssc],
        'hsc_percentage': [hsc],
        'degree_percentage': [degree],
        'technical_score': [technical_score],
        'aptitude_score': [aptitude_score],
        'communication_score': [communication_score],
        'skills_match_percentage': [skills_match],
        'certifications_count': [certifications],
        'internship_experience': [internship],
        'years_of_experience': [experience],
        'career_switch_willingness': [0],
        'relevant_experience': [relevant_exp],
        'previous_ctc_lpa': [5.0],
        'expected_ctc_lpa': [8.0],
        'company_tier': [company_tier],
        'job_role_match': [job_role_match],
        'competition_level': [competition],
        'bond_requirement': [0],
        'notice_period_days': [30.0],
        'layoff_history': [0],
        'employment_gap_months': [0.0],
        'relocation_willingness': [1],
        'degree_specialization_Electronics': [0],
        'degree_specialization_Information Technology': [0],
        'degree_specialization_Mechanical': [0],
        'degree_specialization_Others': [0],
        'experience_category': [1],
        'academic_avg': [(ssc+hsc+degree)/3],
        'academic_performance_bands': [1],
        'skills_match_level': [1],
        'interview_performance_avg': [(technical_score+aptitude_score+communication_score)/3],
        'interview_performance_category': [1],
        'placement_score_raw': [skills_match*0.30 + technical_score*0.25 + aptitude_score*0.25 + communication_score*0.20 + experience*0.05],
        'placement_probability': [0.5],
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result
    if prediction == 1:
        st.success(f"✅ Candidate will be PLACED! Probability: {round(probability*100, 1)}%")
    else:
        st.error(f"❌ Candidate will NOT be placed. Probability: {round(probability*100, 1)}%")