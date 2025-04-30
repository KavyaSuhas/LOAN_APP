import pandas as pd
import numpy as np

# 1. Load raw dataset
df = pd.read_csv("dataset.csv")

# 2. Verify required columns
required_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc', 'loan_status']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# 3. Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 4. Remove duplicate applications
df.drop_duplicates(inplace=True)

# 5. Handle missing values
numeric_columns = ['annual_inc', 'dti', 'fico_range_high', 'open_acc']
for col in numeric_columns:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# 6. Convert interest rate to numeric
if 'int_rate' in df.columns:
    if df['int_rate'].dtype == 'object':  # If it's a string
        df['int_rate'] = df['int_rate'].str.rstrip('%').astype('float') / 100.0
    else:  # If it's already numeric
        df['int_rate'] = df['int_rate'] / 100.0

# 7. Filter loan status to only include relevant categories
if 'loan_status' in df.columns:
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

# 8. Save the cleaned dataset
df.to_csv("cleaned_loan_data.csv", index=False)

print("✅ Cleaned dataset saved as 'cleaned_loan_data.csv'")
print(f"Final dataset shape: {df.shape}")
print(f"Columns in cleaned dataset: {df.columns.tolist()}")

