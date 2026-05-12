import xgboost as xgb
import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = "xgboost_grape_model.json"
DB_PATH = "vineyards_v2.db"


def train_model():
    """Builds a model from the enriched V2 database."""
    print("AI: Training pipeline triggered...")

    if not os.path.exists(DB_PATH):
        print("AI: Database not found. Cannot train.")
        return

    conn = sqlite3.connect(DB_PATH)
    # Ensure we only train on records that actually have the new data
    query = """
    SELECT 
        elevation_GEE_USGS_30m as elevation, 
        slope_GEE_USGS_30m as slope, 
        aspect_GEE_USGS_30m as aspect, 
        hillshade_GEE_USGS_30m as hillshade, 
        mid_year_temp, precipitation, ndvi, ndwi, is_suitable 
    FROM vineyard_features 
    WHERE mid_year_temp IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()


    # CLEANING DATA FOR XGBOOST (Fixes your previous error)
    X = df.drop(columns=['is_suitable'])
    y = df['is_suitable'].astype(int)

    # Force everything to float and fill empty values with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.001, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    print("grid_search")
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    # best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)
    # Get probabilities for the positive class (class 1)
    # y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.4f}")

    print("classification_report")
    print(classification_report(y_test, y_pred))

    print("confusion_matrix")
    print(confusion_matrix(y_test, y_pred))

    # model.fit(X, y)
    best_model.save_model(MODEL_PATH)

    print(f"✅ AI: Model successfully trained on {len(df)} samples and saved.")


def predict_suitability(features: dict) -> bool:
    # AUTO-TRAIN LOGIC
    if not os.path.exists(MODEL_PATH):
        print("AI: Model file not found. Auto-training now...")
        train_model()

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Mapping frontend keys to model keys
    cols = ['elevation', 'slope', 'aspect', 'hillshade', 'mid_year_temp', 'precipitation', 'ndvi', 'ndwi']
    input_values = [float(features.get(c, 0)) for c in cols]

    df_input = pd.DataFrame([input_values], columns=cols)
    prediction = model.predict(df_input)
    return bool(prediction[0])