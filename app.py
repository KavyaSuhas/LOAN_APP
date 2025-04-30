from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import traceback
import sys
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Application directory: {BASE_DIR}")

# Initialize model variables
classifier = None
regressor = None
scaler = None

def load_models():
    global classifier, regressor, scaler
    try:
        model_files = {
            'classifier': "credit_risk_classifier.pkl",
            'regressor': "loan_amount_regressor.pkl",
            'scaler': "scaler.pkl"
        }
        
        logger.info("Checking model files:")
        # Check if files exist
        for model_name, filename in model_files.items():
            # Try multiple possible locations
            possible_paths = [
                os.path.join(BASE_DIR, filename),  # Local development
                os.path.join("/opt/render/project/src", filename),  # Render deployment
                os.path.join("/app", filename),  # Docker deployment
                os.path.join(os.getcwd(), filename)  # Current working directory
            ]
            
            file_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            
            logger.info(f"\nChecking {model_name} model:")
            logger.info(f"  Possible paths: {possible_paths}")
            logger.info(f"  Selected path: {file_path}")
            
            if file_path is None:
                raise FileNotFoundError(f"Model file not found: {filename}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Cannot read model file: {filename}")
            
            logger.info(f"  File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                logger.info(f"  File size: {os.path.getsize(file_path)} bytes")
                logger.info(f"  Readable: {os.access(file_path, os.R_OK)}")
        
        logger.info("\nLoading models:")
        # Load models
        try:
            logger.info("Loading classifier...")
            classifier_path = os.path.join(BASE_DIR, model_files['classifier'])
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
            logger.info("Classifier loaded successfully")
            
            logger.info("Loading regressor...")
            regressor_path = os.path.join(BASE_DIR, model_files['regressor'])
            with open(regressor_path, 'rb') as f:
                regressor = pickle.load(f)
            logger.info("Regressor loaded successfully")
            
            logger.info("Loading scaler...")
            scaler_path = os.path.join(BASE_DIR, model_files['scaler'])
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        
        logger.info("\nAll models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"\nError loading models: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Current directory: {BASE_DIR}")
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return False

# Load models when starting the app
logger.info("\nInitializing application...")
models_loaded = load_models()

@app.route('/')
def home():
    if not models_loaded:
        error_msg = "Error: Model files could not be loaded. Please check the server logs for details."
        logger.error(f"Rendering home page with error: {error_msg}")
        return render_template('index.html', error=error_msg)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        error_msg = "Error: Model files could not be loaded. Please check the server logs for details."
        logger.error(f"Prediction attempted but models not loaded: {error_msg}")
        return render_template('index.html', error=error_msg)
    
    try:
        logger.info("\nReceived prediction request")
        logger.info(f"Form data: {request.form}")
        
        # Get user input
        features = []
        required_fields = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high', 'open_acc']
        
        for field in required_fields:
            value = request.form.get(field)
            logger.info(f"Field {field}: {value}")
            if value is None or value.strip() == '':
                raise ValueError(f"Missing required field: {field}")
            try:
                features.append(float(value))
            except ValueError as e:
                raise ValueError(f"Invalid value for {field}: {value}")
        
        logger.info(f"Features before scaling: {features}")
        try:
            # Convert features to numpy array and reshape for scaling
            features_array = np.array(features).reshape(1, -1)
            logger.info(f"Features array shape: {features_array.shape}")
            scaled = scaler.transform(features_array)
            logger.info(f"Features after scaling: {scaled}")
        except Exception as e:
            logger.error(f"Error in scaling: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise
        
        # Predict risk and loan amount
        try:
            logger.info("Making predictions...")
            risk = classifier.predict(scaled)[0]
            proba = classifier.predict_proba(scaled)[0]
            if len(proba) == 2:
                risk_prob = proba[1]
            else:
                risk_prob = proba[0]
            recommended_amount = regressor.predict(scaled)[0]
            
            logger.info("Prediction results:", {
                'risk': risk,
                'risk_prob': risk_prob,
                'recommended_amount': recommended_amount
            })
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

        return render_template('index.html',
                            prediction=True,
                            risk='High Risk' if risk else 'Low Risk',
                            probability=f"{risk_prob:.2%}",
                            recommendation=f"${recommended_amount:,.2f}")
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(f"\nError in prediction:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return render_template('index.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
