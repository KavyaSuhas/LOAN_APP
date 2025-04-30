import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_data():
    """Create dummy data for training models, ensuring both classes are present"""
    np.random.seed(42)
    n_samples = 1000
    while True:
        # Features matching the Flask app requirements
        loan_amnt = np.random.normal(15000, 5000, n_samples)  # Loan amount
        int_rate = np.random.normal(10, 3, n_samples)  # Interest rate
        annual_inc = np.random.normal(50000, 15000, n_samples)  # Annual income
        dti = np.random.normal(0.3, 0.1, n_samples)  # Debt-to-income ratio
        fico_range_high = np.random.normal(700, 50, n_samples)  # FICO score
        open_acc = np.random.normal(10, 3, n_samples)  # Number of open accounts
        
        X = np.column_stack([
            loan_amnt,
            int_rate,
            annual_inc,
            dti,
            fico_range_high,
            open_acc
        ])
        
        # Credit risk labels (0: low risk, 1: high risk)
        risk_factors = (
            (loan_amnt > 20000) +
            (int_rate > 12) +
            (annual_inc < 45000) +
            (dti > 0.4) +
            (fico_range_high < 650) +
            (open_acc > 15)
        )
        y_risk = (risk_factors >= 3).astype(int)  # High risk if 3 or more risk factors
        
        # Print class distribution
        print("Class distribution:", np.bincount(y_risk))
        if len(np.unique(y_risk)) == 2:
            break  # Both classes present, exit loop
        else:
            print("Regenerating data to ensure both classes are present...")
    
    # Loan amounts (in thousands)
    y_amount = np.random.normal(50, 20, n_samples) * (1 - y_risk) + np.random.normal(20, 10, n_samples) * y_risk
    
    return X, y_risk, y_amount

def train_and_save_models():
    """Train and save models with error handling"""
    try:
        # Create dummy data
        logger.info("Creating dummy data...")
        X, y_risk, y_amount = create_dummy_data()
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        logger.info("Training classifier...")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_scaled, y_risk)
        
        # Train regressor
        logger.info("Training regressor...")
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_scaled, y_amount)
        
        # Save models with compression
        logger.info("Saving models...")
        joblib.dump(classifier, 'credit_risk_classifier.pkl', compress=3)
        joblib.dump(regressor, 'loan_amount_regressor.pkl', compress=3)
        joblib.dump(scaler, 'scaler.pkl', compress=3)
        
        logger.info("Models saved successfully")
        
        # Verify models can be loaded
        logger.info("Verifying models can be loaded...")
        test_load_models()
        
    except Exception as e:
        logger.error(f"Error creating models: {str(e)}")
        raise

def test_load_models():
    """Test loading the saved models"""
    try:
        logger.info("Testing model loading...")
        classifier = joblib.load('credit_risk_classifier.pkl')
        regressor = joblib.load('loan_amount_regressor.pkl')
        scaler = joblib.load('scaler.pkl')
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_and_save_models()
    except Exception as e:
        logger.error(f"Failed to create models: {str(e)}")
        raise 