import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the raw dataset securely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"🔴 Raw data missing at: {file_path}")
    return pd.read_csv(file_path)

def clean_text_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes categorical text data.
    Fixes inconsistencies like 'Male ', 'male', 'MALE' by making everything
    Title Case and stripping trailing or leading whitespaces.
    """
    print("\n🧼 Step 1: Standardizing text formatting...")
    
    # Identify columns containing text/strings
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in string_cols:
        # .str.strip() removes accidental leading/trailing spaces
        # .str.title() forces the text to look like 'Male', 'Tier 1', 'Computer Science'
        df[col] = df[col].astype(str).str.strip().str.title()
        
    print("✅ Text normalization complete.")
    return df

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicate entries.
    Crucial point: Doing this *after* text cleaning catches hidden duplicates
    (e.g., 'Male' and 'Male ' are seen as different until stripped!).
    """
    print("\n👥 Step 2: Checking for duplicate records...")
    
    initial_rows = len(df)
    duplicate_count = df.duplicated().sum()
    print(f"Found {duplicate_count} duplicate rows.")
    
    # drop_duplicates drops exact matching rows and keeps the first occurrence
    df = df.drop_duplicates().reset_index(drop=True)
    
    print(f"🗑️ Removed duplicates. Row count went from {initial_rows} ➡️ {len(df)}")
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in missing (NaN) values using robust, non-leaking statistics:
    - Numerical columns: Filled with Median (middle value, safe from extreme outliers).
    - Categorical columns: Filled with Mode (most frequent value).
    """
    print("\n🩹 Step 3: Checking and handling missing values...")
    
    # 1. Handle Numerical Columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"↳ Imputed numerical column [{col}] using Median: {median_val}")
            
    # 2. Handle Categorical Columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        # Ignore our missing values that were converted to string 'Nan' during title conversion
        df[col] = df[col].replace('Nan', None) 
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"↳ Imputed categorical column [{col}] using Mode: {mode_val}")
            
    print("✅ Missing value imputation complete.")
    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Saves the output cleanly, ensuring the directory structure exists."""
    # os.path.dirname extracts 'data/processed' out of 'data/processed/cleaned_data.csv'
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True) # Creates the folder path if it doesn't exist
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned dataset successfully saved to: {output_path} ({len(df)} rows)")

def main():
    """Orchestrates the entire cleaning pipeline step by step."""
    RAW_DATA_PATH = "data/raw/HR_Job_Placement_Dataset.csv"
    CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"
    
    # Run the processing sequence
    df = load_data(RAW_DATA_PATH)
    df = clean_text_formatting(df)
    df = handle_duplicates(df)
    df = impute_missing_values(df)
    save_processed_data(df, CLEANED_DATA_PATH)

if __name__ == "__main__":
    main()