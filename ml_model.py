# ml_model.py
import os
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path

# Пути должны совпадать с теми, что использует ViticulturePipeline
MODEL_PATH = "xgboost_grape_model.json"
SCALER_PATH = "xgboost_grape_model_scaler.pkl"

# Глобальные объекты для хранения загруженных артефактов
model = None
scaler = None
median_values = None
feature_columns = [
    'elevation_GEE_USGS_30m',
    'slope_GEE_USGS_30m',
    'aspect_GEE_USGS_30m',
    'hillshade_GEE_USGS_30m',
    'mid_year_temp',
    'precipitation',
    'ndvi',
    'ndwi'
]

def load_model_if_exists():
    """Загружает модель и scaler, если файлы существуют. В противном случае model остаётся None."""
    global model, scaler, median_values
    if not os.path.exists(MODEL_PATH):
        print(f"Файл модели {MODEL_PATH} не найден.")
        return

    # Загружаем модель XGBoost
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    # Загружаем scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        raise FileNotFoundError(f"Scaler {SCALER_PATH} не найден, модель неполная.")

    # Медианы заполнения (если не сохраняли, можно вычислить на этапе обучения и сохранить отдельно,
    # но для упрощения здесь можно передавать в функцию predict_suitability уже заполненные данные).
    # В реальном проекте медианы нужно либо сохранить в отдельный файл, либо встроить в пайплайн.
    # Пока оставим None, требуя, чтобы в predict_suitability приходили уже чистые данные.
    median_values = None
    print("Модель и scaler загружены.")

def predict_suitability(env_data: dict) -> bool:
    """
    Предсказание пригодности по словарю с признаками.
    env_data должен содержать ключи, соответствующие feature_columns.
    """
    global model, scaler, median_values
    if model is None:
        raise RuntimeError("Модель не загружена.")

    # Преобразуем словарь в DataFrame
    X = pd.DataFrame([env_data], columns=feature_columns)

    # Заполняем пропуски (если медианы сохранены)
    if median_values is not None:
        X = X.fillna(median_values)

    # Масштабируем
    if scaler is not None:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    else:
        X_scaled = X

    # Предсказание
    pred = model.predict(X_scaled)[0]
    return bool(pred)