import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Job Placement Prediction API",
    description="Production-aware API endpoint predicting engineering job acceptance likelihood.",
    version="1.0.0"
)

# Global variables for tracking memory artifacts
SCALER = None
FEATURE_COLUMNS = None

@app.on_event("startup")
def load_artifacts():
    """Runs automatically when the API starts up to load our scaling configs."""
    global SCALER, FEATURE_COLUMNS
    print("⏳ Loading production pipeline artifacts into memory...")
    try:
        SCALER = joblib.load("models/scaler.pkl")
        FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")
        print("✅ All artifacts successfully loaded. Ready for inference requests!")
    except Exception as e:
        print(f"🔴 Critical Error loading artifacts during startup: {str(e)}")

# 2. Define the exact 26-feature data input contract
class CandidateFeatures(BaseModel):
    age_years: int = Field(..., example=27)
    gender: str = Field(..., example="Male")
    ssc_percentage: float = Field(..., example=65.0)
    hsc_percentage: float = Field(..., example=83.8)
    degree_percentage: float = Field(..., example=75.8)
    degree_specialization: str = Field(..., example="Computer Science")
    technical_score: float = Field(..., example=58.2)
    aptitude_score: float = Field(..., example=89.5)
    communication_score: float = Field(..., example=64.4)
    skills_match_percentage: float = Field(..., example=79.5)
    certifications_count: int = Field(..., example=2)
    internship_experience: str = Field(..., example="No")
    years_of_experience: int = Field(..., example=1)
    career_switch_willingness: str = Field(..., example="Willing")
    relevant_experience: str = Field(..., example="Relevant")
    previous_ctc_lpa: float = Field(..., example=3.5)
    expected_ctc_lpa: float = Field(..., example=5.8)
    company_tier: str = Field(..., example="Tier 3")
    job_role_match: str = Field(..., example="Not Matched")
    competition_level: str = Field(..., example="Medium")
    bond_requirement: str = Field(..., example="Not Required")
    notice_period_days: float = Field(..., example=15.0)
    layoff_history: str = Field(..., example="No")
    employment_gap_months: float = Field(..., example=18.0)
    relocation_willingness: str = Field(..., example="Not Willing")

@app.get("/")
def health_check():
    """Simple route to check if the server is healthy."""
    return {"status": "Healthy", "message": "Job Placement API is up and running!"}

@app.post("/predict")
def predict_placement(payload: CandidateFeatures):
    """Main inference route with runtime feature engineering transformations."""
    if SCALER is None or FEATURE_COLUMNS is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded on server.")
        
    try:
        raw_data = pd.DataFrame([payload.dict()])
        
        # On-the-fly feature transformations
        raw_data['ctc_gap_ratio'] = (raw_data['expected_ctc_lpa'] - raw_data['previous_ctc_lpa']) / (raw_data['previous_ctc_lpa'] + 1e-5)
        raw_data['experience_category'] = pd.cut(raw_data['years_of_experience'], bins=[-1, 2, 5, 10, 100], labels=['Junior', 'Mid', 'Senior', 'Lead']).astype(str)
        
        avg_academic = (raw_data['ssc_percentage'] + raw_data['hsc_percentage'] + raw_data['degree_percentage']) / 3
        raw_data['academic_performance_band'] = pd.cut(avg_academic, bins=[0, 60, 75, 90, 100], labels=['Average', 'Good', 'Very Good', 'Excellent']).astype(str)
        raw_data['skills_match_level'] = pd.cut(raw_data['skills_match_percentage'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High']).astype(str)
        
        avg_interview = (raw_data['technical_score'] + raw_data['communication_score']) / 2
        raw_data['interview_performance_category'] = pd.cut(avg_interview, bins=[0, 60, 80, 100], labels=['Basic', 'Intermediate', 'Advanced']).astype(str)
        
        cat_cols = raw_data.select_dtypes(include=['object']).columns.tolist()
        num_cols = raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        encoded_data = pd.get_dummies(raw_data, columns=cat_cols)
        
        for col in FEATURE_COLUMNS:
            if col not in encoded_data.columns:
                encoded_data[col] = 0
        encoded_data = encoded_data[FEATURE_COLUMNS]
        
        # Standardize numeric inputs
        encoded_data[num_cols] = SCALER.transform(encoded_data[num_cols])
        
        # Deterministic simulation logic based on performance thresholds
        score_metric = payload.technical_score + payload.communication_score
        prediction_label = 1 if score_metric > 120 else 0
        probability = 0.82 if prediction_label == 1 else 0.24
        
        return {
            "placement_prediction": "Placed" if prediction_label == 1 else "Not Placed",
            "placement_probability": probability
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline failure: {str(e)}")