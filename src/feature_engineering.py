import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Loads our pristine cleaned data from the previous pipeline step."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"🔴 Cleaned data missing at: {file_path}. Run data_cleaning.py first!")
    return pd.read_csv(file_path)

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers 5 brand new domain-specific features from raw columns
    to make the data more informative for our ML models.
    """
    print("\n🧠 Step 1: Engineering derived features...")
    
    # 1. ctc_gap_ratio: Ratio of expected package vs what they currently make
    # Avoid zero division by adding a tiny epsilon if previous_ctc is 0
    df['ctc_gap_ratio'] = (df['expected_ctc_lpa'] - df['previous_ctc_lpa']) / (df['previous_ctc_lpa'] + 1e-5)
    
    # 2. experience_category: Convert continuous years into clear career buckets
    df['experience_category'] = pd.cut(
        df['years_of_experience'], 
        bins=[-1, 2, 5, 10, 100], 
        labels=['Junior', 'Mid', 'Senior', 'Lead']
    ).astype(str)
    
    # 3. academic_performance_band: Average of school and degree scores to find top academic tiers
    df['academic_performance_band'] = (df['ssc_percentage'] + df['hsc_percentage'] + df['degree_percentage']) / 3
    df['academic_performance_band'] = pd.cut(
        df['academic_performance_band'],
        bins=[0, 60, 75, 90, 100],
        labels=['Average', 'Good', 'Very Good', 'Excellent']
    ).astype(str)
    
    # 4. skills_match_level: Simple category grouping for percentage scores
    df['skills_match_level'] = pd.cut(
        df['skills_match_percentage'],
        bins=[0, 50, 75, 100],
        labels=['Low', 'Medium', 'High']
    ).astype(str)
    
    # 5. interview_performance_category: Combines technical and communication scores
    total_interview_score = (df['technical_score'] + df['communication_score']) / 2
    df['interview_performance_category'] = pd.cut(
        total_interview_score,
        bins=[0, 60, 80, 100],
        labels=['Basic', 'Intermediate', 'Advanced']
    ).astype(str)
    
    print("✅ Derived features created successfully.")
    return df

def split_dataset(df: pd.DataFrame):
    """
    Splits the data into Train (70%), Validation (15%), and Test (15%).
    Uses stratification on our target ('status') to maintain the 70/30 class imbalance.
    """
    print("\n✂️ Step 2: Splitting dataset into Train/Val/Test...")
    
    # Target variable 'status' needs to be mapped to binary integers for the models
    df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
    
    # Split out the features (X) and target (y)
    X = df.drop(columns=['status'])
    y = df['status']
    
    # Split 1: Split into Train (70%) and a temporary set (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Split 2: Split the temporary 30% exactly in half into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"↳ Train shape: {X_train.shape} | Val shape: {X_val.shape} | Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_and_scale(X_train, X_val, X_test):
    """
    FITS scalers and encoders ONLY on Train to completely prevent data leakage.
    Uses those fitted artifacts to transform Val and Test.
    """
    print("\n🛡️ Step 3: Enforcing leakage protection, encoding, and scaling...")
    
    # Separate columns by data types
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Initialize dictionary containers to store our fitted models
    saved_encoders = {}
    
    # 1. Process Categorical Columns using One-Hot Encoding via pandas get_dummies
    # We do a high-quality trick: align columns so all splits match exactly
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Ensure all splits have the exact same columns after one-hot encoding
    X_train_encoded, X_val_encoded = X_train_encoded.align(X_val_encoded, join='left', axis=1, fill_value=0)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    # Save the exact list of final columns for our production FastAPI app schema matching
    feature_columns = X_train_encoded.columns.tolist()
    
    # 2. Fit standard scaler ONLY on training numeric data
    scaler = StandardScaler()
    
    # We modify the copies directly to keep pandas formatting clean
    X_train_final = X_train_encoded.copy()
    X_val_final = X_val_encoded.copy()
    X_test_final = X_test_encoded.copy()
    
    X_train_final[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_val_final[num_cols] = scaler.transform(X_val_encoded[num_cols])
    X_test_final[num_cols] = scaler.transform(X_test_encoded[num_cols])
    
    # Create models directory if it doesn't exist to store our artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    
    print("✅ Scaling complete. Encoders and scalers securely backed up to 'models/'.")
    return X_train_final, X_val_final, X_test_final

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Recombines X and y to save pristine CSV files for Phase 2 training."""
    print("\n💾 Step 4: Saving engineered splits to disk...")
    
    # Add target column back to its respective split
    train_full = X_train.copy()
    train_full['status'] = y_train
    
    val_full = X_val.copy()
    val_full['status'] = y_val
    
    test_full = X_test.copy()
    test_full['status'] = y_test
    
    # Save files
    train_full.to_csv("data/processed/train.csv", index=False)
    val_full.to_csv("data/processed/val.csv", index=False)
    test_full.to_csv("data/processed/test.csv", index=False)
    
    print("🎉 All files written! Prepared data/processed/ {train.csv, val.csv, test.csv}")

def main():
    cleaned_path = "data/processed/cleaned_data.csv"
    
    df = load_cleaned_data(cleaned_path)
    df = create_derived_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled = transform_and_scale(X_train, X_val, X_test)
    save_splits(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

if __name__ == "__main__":
    main()