import os
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

# Locate paths relative to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_grape_model.json")
SCALER_PATH = os.path.join(BASE_DIR, "xgboost_grape_model_scaler.pkl")

model = None
scaler = None
median_values = None
feature_columns = [
    'elevation_GEE_USGS_30m',
    'slope_GEE_USGS_30m',
    'aspect_GEE_USGS_30m',
    'hillshade_GEE_USGS_30m',
    'mid_year_temp', 'precipitation', 'ndvi', 'ndwi',
    'solar_radiation', 'humidity', 'wind_speed',
    'evapotranspiration', 'evi', 'lai', 'soil_ph',
    'soil_organic_carbon', 'fire_risk'
]


def load_model_if_exists():
    global model, scaler, median_values
    if not os.path.exists(MODEL_PATH):
        print(f"Файл модели {MODEL_PATH} не найден.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        raise FileNotFoundError(f"Scaler {SCALER_PATH} не найден, модель неполная.")

    median_values = None
    print("Модель и scaler загружены.")


def predict_suitability(env_data: dict) -> bool:
    global model, scaler, median_values
    if model is None:
        raise RuntimeError("Модель не загружена.")

    X = pd.DataFrame([env_data], columns=feature_columns)

    if median_values is not None:
        X = X.fillna(median_values)

    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    else:
        X_scaled = X

    pred = model.predict(X_scaled)[0]
    return bool(pred)


class GrapeXGBClassifier:
    loaded_model = None
    loaded_scaler = None
    loaded_feature_names = None
    loaded_target_names = None

    def __init__(self, path_to_model=None):
        print('1 load_model or train model dataset in pd.DataFrame format needed')
        self.load_model(path_to_model)
        print('2 predict')

    def load_model(self, path_to_model=None):
        import joblib
        import pandas as pd
        import numpy as np
        import os

        print("\nLoading the trained model, scaler, and metadata...")
        if path_to_model:
            model_dir = path_to_model
        else:
            model_dir = os.path.join(BASE_DIR, 'trained_models')

        model_filename = os.path.join(model_dir, 'multi_output_xgb_model.joblib')
        scaler_filename = os.path.join(model_dir, 'scaler.joblib')
        feature_names_filename = os.path.join(model_dir, 'feature_names.joblib')
        target_names_filename = os.path.join(model_dir, 'target_names.joblib')

        try:
            self.loaded_model = joblib.load(model_filename)
            self.loaded_scaler = joblib.load(scaler_filename)
            self.loaded_feature_names = joblib.load(feature_names_filename)
            self.loaded_target_names = joblib.load(target_names_filename)
            print("Model, scaler, and metadata loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model files not found in {model_dir}. Please ensure the training script was run.")
            exit()

    def save_model(self, multi_output_xgb_model, scaler_multi, feature_names, target_names):
        import joblib
        import os

        print("\nSaving the trained model...")
        model_dir = os.path.join(BASE_DIR, 'trained_models')
        model_filename = os.path.join(model_dir, 'multi_output_xgb_model.joblib')

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(multi_output_xgb_model, model_filename)
        print(f"Model saved successfully to {model_filename}")

        scaler_filename = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(scaler_multi, scaler_filename)
        print(f"Scaler saved successfully to {scaler_filename}")

        feature_names_filename = os.path.join(model_dir, 'feature_names.joblib')
        joblib.dump(feature_names, feature_names_filename)
        print(f"Feature names saved successfully to {feature_names_filename}")

        target_names_filename = os.path.join(model_dir, 'target_names.joblib')
        joblib.dump(target_names, target_names_filename)
        print(f"Target names saved successfully to {target_names_filename}")

    def train(self, df):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.metrics import f1_score, roc_auc_score, jaccard_score

        feature_names = ['elevation', 'slope', 'aspect', 'hillshade', 'mid_year_temp', 'precipitation', 'ndvi', 'ndwi']
        target_names = ['Arnsburger', 'Arinto', 'Mostosa', 'Abbuoto', 'Abouriou', 'Acitana']

        X = df[feature_names].apply(pd.to_numeric)
        y = df[target_names].apply(pd.to_numeric)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        print("\nScaling features...")
        scaler_multi = StandardScaler()
        X_train_scaled = scaler_multi.fit_transform(X_train)
        X_test_scaled = scaler_multi.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        print("\nTraining XGBoost MultiOutputClassifier...")
        xgb_base_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1
        )

        multi_output_xgb_model = MultiOutputClassifier(xgb_base_model, n_jobs=-1)
        multi_output_xgb_model.fit(X_train_scaled_df, y_train)

        print("XGBoost MultiOutputClassifier training complete.")
        print("\nEvaluating the model...")

        y_pred_proba = multi_output_xgb_model.predict_proba(X_test_scaled_df)
        y_pred_proba_reshaped = np.array([prob[:, 1] for prob in y_pred_proba]).T
        y_pred_binary = (y_pred_proba_reshaped > 0.5).astype(int)

        f1_micro = f1_score(y_test, y_pred_binary, average='micro')
        f1_macro = f1_score(y_test, y_pred_binary, average='macro')
        jaccard = jaccard_score(y_test, y_pred_binary, average='samples')

        roc_auc_scores = []
        for i, target_name in enumerate(target_names):
            if len(np.unique(y_test.iloc[:, i])) > 1:
                roc_auc_scores.append(roc_auc_score(y_test.iloc[:, i], y_pred_proba_reshaped[:, i]))
            else:
                print(f"Skipping ROC AUC for {target_name} as it has only one class in y_test.")

        roc_auc_macro = np.mean(roc_auc_scores) if roc_auc_scores else np.nan

        print(f"\n--- Evaluation Results ---")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Jaccard Score (Samples Avg): {jaccard:.4f}")
        print(f"ROC AUC Score (Macro Avg): {roc_auc_macro:.4f}")

        print("\nPer-label F1 Scores:")
        f1_per_label = f1_score(y_test, y_pred_binary, average=None)
        for i, target_name in enumerate(target_names):
            print(f"  {target_name}: {f1_per_label[i]:.4f}")

        print("\nTrain finished.")

    def predict(self, new_data_raw):
        if not self.loaded_model:
            print('model not loaded')
            return None
        if not self.loaded_scaler:
            print('scaler not loaded')
            return None
        if not self.loaded_feature_names:
            print('feature names not loaded')
            return None
        if not self.loaded_target_names:
            print('target names not loaded')
            return None

        print("\nMaking predictions with the loaded model and ranking results...")

        global median_values
        new_data_raw = pd.DataFrame([new_data_raw], columns=feature_columns)

        if median_values is not None:
            new_data_raw = new_data_raw.fillna(median_values)

        new_data_raw.rename(columns={
            'aspect_GEE_USGS_30m': 'aspect',
            'elevation_GEE_USGS_30m': 'elevation',
            'hillshade_GEE_USGS_30m': 'hillshade',
            'slope_GEE_USGS_30m': 'slope',
        }, inplace=True)

        new_data_scaled = self.loaded_scaler.transform(new_data_raw)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=self.loaded_feature_names)

        sample_pred_probas = self.loaded_model.predict_proba(new_data_scaled_df)
        probabilities_for_targets = [float(p[0][1]) for p in sample_pred_probas]

        target_probs_dict = dict(zip(self.loaded_target_names, probabilities_for_targets))
        ranked_targets = sorted(target_probs_dict.items(), key=lambda item: item[1], reverse=True)

        rankings = []
        for tup_grape in ranked_targets:
            rankings.append({"grape": tup_grape[0], "score": round(tup_grape[1] * 100, 2)})

        return rankings