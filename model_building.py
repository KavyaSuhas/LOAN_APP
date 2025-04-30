import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------
# 1. Load Cleaned Dataset
# ----------------------------------------
df = pd.read_csv("cleaned_loan_data.csv")

# Ensure all required columns are present
required_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc', 'loan_status']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert numeric columns to appropriate types
numeric_columns = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter target classes and convert to numeric
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['loan_status_encoded'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# ----------------------------------------
# 2. Feature Selection
# ----------------------------------------
features = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc']
X = df[features]
y_class = df['loan_status_encoded']
y_reg = df['loan_amnt']  # Same as a feature, but just for regression demo

# ----------------------------------------
# 3. Train/Test Split
# ----------------------------------------
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# ----------------------------------------
# 4. Classification Model – XGBoost
# ----------------------------------------
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_class_train)
y_pred_class = xgb_clf.predict(X_test)

print("\n📊 Classification Report (XGBoost):")
print(classification_report(y_class_test, y_pred_class))

# ----------------------------------------
# 5. Regression Model – LightGBM
# ----------------------------------------
lgb_reg = lgb.LGBMRegressor()
lgb_reg.fit(X_train, y_reg_train)
y_pred_reg = lgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))

print(f"\n📉 RMSE (Loan Amount Prediction): {rmse:.2f}")

# ----------------------------------------
# 6. Clustering – KMeans for Segmentation
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
print("\n👥 Applicant Clustering Summary:")
print(df['cluster'].value_counts())

# ----------------------------------------
# 7. Save Models
# ----------------------------------------
joblib.dump(xgb_clf, "credit_risk_classifier.pkl")
joblib.dump(lgb_reg, "loan_amount_regressor.pkl")
joblib.dump(kmeans, "applicant_cluster_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models saved successfully!")
