import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

DB_PATH = "vineyard_features.db"
MODEL_PATH = "xgboost_grape_model.json"


def load_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    # Pull the specific terrain columns
    query = """
    SELECT 
        elevation_GEE_USGS_30m as elevation, slope_GEE_USGS_30m as slope, 
        aspect_GEE_USGS_30m as aspect, hillshade_GEE_USGS_30m as hillshade, 
        mid_year_temp, precipitation, ndvi, ndwi, is_suitable 
    FROM vineyard_features
    WHERE mid_year_temp IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def add_synthetic_negative_samples(df, num_negatives=500):
    """
    Since the DB currently only has valid vineyards (Positives),
    we must generate some bad locations (Negatives) so the AI can learn the difference.
    """
    print(f"Generating {num_negatives} synthetic NEGATIVE examples for contrast...")
    np.random.seed(42)

    # Generate data that is generally bad for grapes
    # (e.g., extremely high elevation, completely flat or overly steep slopes, very low sunlight/hillshade)
    neg_df = pd.DataFrame({
        'elevation': np.random.uniform(1000, 3000, num_negatives),  # Too high
        'slope': np.concatenate([np.random.uniform(0, 1, int(num_negatives / 2)),  # Too flat
                                 np.random.uniform(30, 60, int(num_negatives / 2))]),  # Too steep
        'aspect': np.random.uniform(0, 360, num_negatives),
        'hillshade': np.random.uniform(0, 100, num_negatives),  # Too dark/shaded
        'is_suitable': 0  # 0 = False
    })

    return pd.concat([df, neg_df], ignore_index=True)


def run_training_pipeline():
    print("--- Starting Grape Vine Model Training ---")

    # 1. Load Real Data
    df = load_data_from_db()
    print(f"Loaded {len(df)} real vineyards from database.")

    # 2. Check for negative samples (zeros). If none, add synthetic ones.
    if 0 not in df['is_suitable'].values:
        df = add_synthetic_negative_samples(df, num_negatives=max(len(df), 500))

    # 3. Prepare features and target
    X = df[['elevation', 'slope', 'aspect', 'hillshade', 'mid_year_temp', 'precipitation', 'ndvi', 'ndwi']]
    y = df['is_suitable']

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Model Setup & Grid
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    print("\nStarting Hyperparameter Tuning...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, param_distributions=param_grid,
        n_iter=10, scoring='accuracy', cv=3, random_state=42, n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    print("\n--- Model Evaluation ---")
    y_pred = best_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save for the web app to use
    best_model.save_model(MODEL_PATH)
    print(f"✅ Optimized model saved to '{MODEL_PATH}'!")


if __name__ == "__main__":
    run_training_pipeline()