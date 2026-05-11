import xgboost as xgb
import pandas as pd
import numpy as np
import os

MODEL_PATH = "xgboost_grape_model.json"


def train_initial_model():
    """Trains a synthetic model if one doesn't exist yet."""
    # Synthetic data for initialization
    np.random.seed(42)
    n_samples = 200

    # Features: temp (15-20C is good), precip, frost_risk, elevation, ndvi, ndwi
    X = pd.DataFrame({
        'mid_year_temp': np.random.uniform(10, 25, n_samples),
        'precipitation': np.random.uniform(300, 1000, n_samples),
        'frost_risk': np.random.uniform(0, 1, n_samples),
        'elevation': np.random.uniform(50, 800, n_samples),
        'ndvi': np.random.uniform(0.2, 0.8, n_samples),
        'ndwi': np.random.uniform(-0.5, 0.5, n_samples)
    })

    # Simple logic to determine suitability for the dummy model
    y = ((X['mid_year_temp'] > 14) & (X['mid_year_temp'] < 22) &
         (X['frost_risk'] < 0.3) & (X['elevation'] < 500)).astype(int)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    model.save_model(MODEL_PATH)
    print("Initial XGBoost model trained and saved.")


def predict_suitability(features: dict) -> bool:
    if not os.path.exists(MODEL_PATH):
        train_initial_model()

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Convert dict to DataFrame
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return bool(prediction[0])