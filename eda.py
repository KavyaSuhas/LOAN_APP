# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting style
sns.set(style='whitegrid', palette='muted')
plt.rcParams["figure.figsize"] = (10, 6)

# Load dataset
df = pd.read_csv('cleaned_loan_data.csv')  # Replace with actual path

# 1. Data Overview
print("Data Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# 2. Default patterns by demographic
plt.figure()
sns.countplot(x='home_ownership', hue='loan_status', data=df)
plt.title('Loan Status by Home Ownership')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("loan_status_by_home_ownership.png")

plt.figure()
sns.countplot(x='purpose', hue='loan_status', data=df, order=df['purpose'].value_counts().index)
plt.title('Loan Status by Loan Purpose')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig("loan_status_by_loan_purpose.png")

# 3. Key Risk Factor Correlations
numerical_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc']
corr = df[numerical_cols].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Key Financial Features")
plt.tight_layout()
plt.show()
plt.savefig("correlation_matrix_of_key_financial_features.png")

# 4. Default Rate by FICO Score
plt.figure()
sns.histplot(data=df, x='fico_range_high', hue='loan_status', bins=30, kde=True, element='step')
plt.title("FICO Score vs Loan Status")
plt.tight_layout()
plt.show()
plt.savefig("fico_score_vs_loan_status.png")
# 5. Loan Performance over Time
if 'issue_d' in df.columns:
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    df['issue_month'] = df['issue_d'].dt.to_period('M')
    
    plt.figure()
    df.groupby('issue_month')['loan_status'].value_counts(normalize=True).unstack().plot(kind='line')
    plt.title("Loan Default Trends Over Time")
    plt.ylabel("Proportion")
    plt.xlabel("Issue Date")
    plt.tight_layout()
    plt.show()
    plt.savefig("loan_default_trends_over_time.png")
else:
    print("Column 'issue_d' not found — skipping time series analysis.")

