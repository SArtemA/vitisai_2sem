import os
import sqlite3
import sys

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join("databases", "grapes_annotated_v2.db")
MODEL_DIR = os.path.join("models", "trained_models")

FEATURE_COLUMNS = [
    "elevation",
    "slope",
    "aspect",
    "hillshade",
    "mid_year_temp",
    "precipitation",
    "ndvi",
    "ndwi",
    "solar_radiation",
    "humidity",
    "wind_speed",
    "evapotranspiration",
    "evi",
    "lai",
    "land_cover_type",
    "soil_ph",
    "soil_organic_carbon",
    "fire_risk",
    "winkler_index",
    "fpar",
    "surface_pressure",
    "potential_evaporation_sum",
    "surface_sensible_heat_flux_sum",
    "volumetric_soil_water_layer_3",
    "skin_reservoir_content",
    "soil_temperature_level_3",
    "skin_temperature",
    "soil_bulk_density",
    "soil_sand",
    "soil_clay",
    "soil_texture_class",
]
MUL_TARGET = ["arnsburger", "arinto", "mostosa", "abbuoto", "abouriou", "acitana"]


def retrain_pipeline():
    if not os.path.exists(DB_PATH):
        print(f"Ошибка: База данных '{DB_PATH}' не найдена.")
        return False

    print("Загрузка данных из БД...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM annotated_grapes;", conn)
    conn.close()

    if len(df) < 50:
        print("Недостаточно данных для обучения (менее 50 записей). Отмена.")
        return False

    # Определяем целевые колонки сортов (все колонки кроме служебных и признаков)
    # excluded = set(FEATURE_COLUMNS + ['id', 'lat', 'lon'])
    target_columns = MUL_TARGET

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[target_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Обучение новой модели
    print("Обучение новой модели XGBoost...")
    base_model = XGBClassifier(
        colsample_bytree=0.7,
        gamma=0.2,
        learning_rate=0.1,
        max_depth=3,
        n_estimators=300,
        reg_alpha=0.001,
        reg_lambda=1,
        subsample=0.7,
        random_state=42,
        n_jobs=-1,
    )
    new_model = MultiOutputClassifier(base_model, n_jobs=-1)
    new_model.fit(X_train_scaled, y_train)

    # 3. Оценка качества новой модели
    # y_pred_proba возвращает список массивов по выходам:
    import numpy as np

    y_pred_probs = np.array([prob[:, 1] for prob in new_model.predict_proba(X_test_scaled)]).T
    new_f1 = f1_score(y_test, (y_pred_probs > 0.5).astype(int), average="micro")
    print(f"Новая модель: Micro F1-score = {new_f1:.4f}")

    # 4. Проверка качества старой модели (Gatekeeper)
    old_model_path = os.path.join(MODEL_DIR, "multi_output_xgb_model.joblib")
    old_scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if os.path.exists(old_model_path) and os.path.exists(old_scaler_path):
        try:
            old_model = joblib.load(old_model_path)
            old_scaler = joblib.load(old_scaler_path)
            X_test_old_scaled = old_scaler.transform(X_test)
            old_probs = np.array([prob[:, 1] for prob in old_model.predict_proba(X_test_old_scaled)]).T
            old_f1 = f1_score(y_test, (old_probs > 0.5).astype(int), average="micro")
            print(f"Текущая модель: Micro F1-score = {old_f1:.4f}")

            if new_f1 < old_f1:
                print("Качество новой модели ниже текущей. Замена отменена.")
                return False
        except Exception as e:
            print(f"Не удалось сравнить со старой моделью ({e}). Замена будет выполнена.")

    # 5. Сохранение артефактов
    print(f"Сохранение обновленной модели в {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(new_model, os.path.join(MODEL_DIR, "multi_output_xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(target_columns, os.path.join(MODEL_DIR, "target_names.joblib"))

    print("Модель успешно обновлена.")
    return True


if __name__ == "__main__":
    retrain_pipeline()
