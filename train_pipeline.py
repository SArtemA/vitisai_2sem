import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

DB_PATH = "./vineyards.db"
MODEL_PATH = "xgboost_grape_model.json"
MIN_SAMPLES_REQUIRED = 50


def load_data_from_db():
    """Loads data from the SQLite database."""
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT mid_year_temp, precipitation, frost_risk, elevation, ndvi, ndwi, is_suitable FROM vineyards"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def generate_synthetic_data(n_samples=500):
    """Generates synthetic data if the database doesn't have enough rows yet."""
    print(f"Generating {n_samples} synthetic samples for robust training...")
    np.random.seed(42)

    # Generate random features
    df = pd.DataFrame({
        'mid_year_temp': np.random.uniform(5, 35, n_samples),
        'precipitation': np.random.uniform(100, 1500, n_samples),
        'frost_risk': np.random.uniform(0, 1, n_samples),
        'elevation': np.random.uniform(0, 1200, n_samples),
        'ndvi': np.random.uniform(0.0, 1.0, n_samples),
        'ndwi': np.random.uniform(-1.0, 1.0, n_samples)
    })

    # Define viticulture suitability rules to create realistic labels
    # Ideal: Temp 14-22C, Precip 400-800mm, Frost Risk < 0.3, Elev < 600m
    conditions = (
            (df['mid_year_temp'] >= 14) & (df['mid_year_temp'] <= 22) &
            (df['precipitation'] >= 400) & (df['precipitation'] <= 800) &
            (df['frost_risk'] < 0.3) &
            (df['elevation'] < 600) &
            (df['ndvi'] > 0.3)
    )

    # Add some noise/randomness so the model has to learn probabilities
    df['is_suitable'] = np.where(conditions, 1, 0)

    # Flip ~5% of labels to add noise
    noise_indices = df.sample(frac=0.05).index
    df.loc[noise_indices, 'is_suitable'] = 1 - df.loc[noise_indices, 'is_suitable']

    return df


def run_training_pipeline():
    print("--- Starting Grape Vine Model Training Pipeline ---")

    # 1. Load Data
    df = load_data_from_db()
    if len(df) < MIN_SAMPLES_REQUIRED:
        print(f"Only {len(df)} records found in DB. Not enough for fine-tuning.")
        df = generate_synthetic_data(n_samples=1000)
    else:
        print(f"Loaded {len(df)} records from the database.")

    # 2. Prepare features and target
    X = df.drop(columns=['is_suitable'])
    y = df['is_suitable']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # 4. Define XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # 5. Define Hyperparameter Grid
    # These are the parameters we will test to find the optimal combination
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5]
    }

    # 6. Setup RandomizedSearchCV for Fine-Tuning
    # (Using RandomizedSearch instead of GridSearch to save time while finding great params)
    print("\nStarting Hyperparameter Tuning (this may take a moment)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings that are sampled
        scoring='accuracy',  # Optimize for accuracy
        cv=3,  # 3-fold cross validation
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # 7. Fit the model
    random_search.fit(X_train, y_train)

    # 8. Extract the best model
    best_model = random_search.best_estimator_
    print(f"\nBest Hyperparameters found:\n{random_search.best_params_}")

    # 9. Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 10. Save the optimized model
    best_model.save_model(MODEL_PATH)
    print(f"--- Pipeline Complete! Optimized model saved to '{MODEL_PATH}' ---")
    print("The web app will automatically use this new model for future predictions.")


if __name__ == "__main__":
    run_training_pipeline()